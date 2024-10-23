import os
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from collections import OrderedDict, defaultdict
from .layers import LongRangeModulation

__all__ = ['LRMNet', 'SteerableLRM']
        
class LRMNet(nn.Module):
    def __init__(self, network, mod_connections, forward_passes=2, img_size=224, 
                 return_list=True):
        super().__init__()
        # mod_connections defines which layers will receive feedback from other layers
        self.forward_passes = forward_passes
        self.feedforward = network # Core feed-forward network
        self.prevent_inplace_after_connections(mod_connections)
        self.return_list = return_list
        
        # Create Modulatory Connections
        connections = self.group_by_destination(mod_connections)
        # These connections are grouped by destination using group_by_destination(), 
        # and for each destination layer, a LongRangeModulation block is created, which manages the feedback process.
        mod_layers = OrderedDict([])
        for dest_layer, source_layers in connections.items():
            # here, it registers hooks on specific layers of the network (both the source and destination layers) 
            # to capture activations during forward passes.
            mod_layer = LongRangeModulation(network, dest_layer, source_layers, img_size=img_size)
            mod_layers[mod_layer.name] = mod_layer
        
        # These blocks are stored in self.lrm, which is a sequential container of all feedback connections.
        self.lrm = nn.Sequential(mod_layers)
        
        # Clear mod_inputs created when initilizing modules
        for block in self.lrm.children(): 
            block.mod_inputs = {}
                
    def group_by_destination(self, mod_connections):
        connections = defaultdict(list)
        for conn in mod_connections:
            connections[conn['destination']].append(conn['source'])

        return connections        
    
    # To ensure proper handling of feedback, any in-place operations in the network’s layers 
    # that could interfere with feedback are disabled.
    def prevent_inplace_after_connections(self, mod_connections):
        # Extract all layers from source and destination
        layers_to_check = set()

        for conn in mod_connections:
            for key in ['source', 'destination']:
                layers_to_check.add(conn[key])

        # Flag to check if the previous layer was a source or destination
        prev_layer_in_connections = False

        # Loop over named modules
        for name, module in self.feedforward.named_modules():
            if prev_layer_in_connections and hasattr(module, 'inplace'):
                module.inplace = False
            prev_layer_in_connections = name in layers_to_check
    
    def forward(self, x, drop_state=True, forward_passes=None, return_list=None):
        '''
            forward pass for LRMs
            
            Default behavior is to drop any stored activations, perform the default number
            of forward_passes, and return the output of the final forward pass.
            
            However, there are several possible overrides that can be enabled.
            
            drop_state can be set to False to retain prior modulatory activation states,
            and the number of forward_passes can be specified. This could be useful
            if you wanted to manually step through N forwarded passes,
                out1 = model(x, drop_state=True, forward_passes=1)
                out2 = model(x, drop_state=False, forward_passes=1)
                out3 = model(x, drop_state=False, forward_passes=1)
                ...
            
            It's also possible to return the outputs of each forward pass (list_outputs=True).                                                     
        '''
        # If drop_state is True, any stored feedback activations are cleared before each pass, ensuring fresh feedback is applied.
        return_list = self.return_list if return_list is None else return_list
        
        # Optionally drop any stored feedback or skip inputs (from previous batch)
        if drop_state:
            for block in self.lrm.children(): 
                block.mod_inputs = {}

        # iterate over forward_passes,
        # - input-drive is always the same (x)
        # - first pass not modulated by feedback (unless first_pass_steering)
        # - all subsequent passes modulated by feedback        
        outputs = []
        forward_passes = self.forward_passes if forward_passes is None else forward_passes
        for pass_num in range(0, forward_passes):
            # The model doesn’t directly distinguish activations by pass number.
            out = self.feedforward(x)
            outputs.append(out)

        if return_list:
            return outputs
        
        return out

# a bit more info:
'''
First Pass:
•	The model performs a regular forward pass.
•	Activations from the source layers are captured using hooks and stored in mod_inputs.
•	No feedback is applied in the first pass.
Subsequent Passes:
•	The stored activations in mod_inputs are used to modulate the activations in the target layers.
•	After each pass, the feedback activations are updated to reflect the new activations from the deeper layers.
'''

class SteerableLRM(nn.Module):
    def __init__(self, network, mod_connections, forward_passes=2, return_list=True, 
                 img_size=224):
        super().__init__()
        self.forward_passes = forward_passes
        self.feedforward = network
        self.return_list = self.return_list if return_list is None else return_list
        
        # Create Modulatory Connections
        connections = self.group_by_destination(mod_connections)
        mod_layers = OrderedDict([])
        for dest_layer, source_layers in connections.items():
            mod_layer = LongRangeModulation(network, dest_layer, source_layers, img_size=img_size)
            mod_layers[mod_layer.name] = mod_layer
            
        self.lrm = nn.Sequential(mod_layers)
        
        # Clear mod_inputs created when initilizing modules
        for block in self.lrm.children(): 
            block.mod_inputs = {}
                
    def group_by_destination(self, mod_connections):
        connections = defaultdict(list)
        for conn in mod_connections:         
            connections[conn['destination']].append(conn['source'])

        return connections        
    

    def add_steering_signals(self, steering_signals):
        for steering_signal in steering_signals:
            self.add_steering_signal(**steering_signal)

    # This method applies the external steering signal by injecting it into the feedback mechanism, 
    # replacing or blending it with the native feedback activations.  
    # Strengths are the positive/negative scalers learned
    # Alpha is a weighted factor: controls how much of the steering signal is blended with the original feedback from the network.   
    def add_steering_signal(self, source, activation, strength, alpha=1):
        neg_scale_adjust, pos_scale_adjust = _pair(strength) # Split strength into neg/pos scale adjustments
        
        # find modules targeted by this steering_source, replace their
        # modulatory inputs for this source with steering activation
        pattern = f'from_{source.replace(".","_")}_to_'
        for lrm_module_name,lrm_module in self.lrm.named_children():            
            for feedback_module_name,feedback_module in lrm_module.named_children():                                
                # If this ModBlock is from steering source to current lrm_module, add steering
                if pattern in feedback_module_name:  
                    # adjust modulation strengths
                    self.adjust_modulation_strengths(feedback_module, neg_scale_adjust, pos_scale_adjust)
                
                    if feedback_module_name in lrm_module.mod_inputs:
                        # update current modulatory input (if alpha==1, we end up just replacing it with steering activation)
                        curr_input = lrm_module.mod_inputs[feedback_module_name]
                        lrm_module.mod_inputs[feedback_module_name] = alpha * activation + (1-alpha) * curr_input

                    else:
                        # without curr_input, simply set to template
                        lrm_module.mod_inputs[feedback_module_name] = activation
        
    @torch.no_grad()
    # This method adjusts the strength of the steering signal, 
    # controlling how much it amplifies or dampens the feedback.
    def adjust_modulation_strengths(self, feedback_module, neg_scale_adjust, pos_scale_adjust):
        if neg_scale_adjust != 1.0 and not hasattr(feedback_module, 'neg_scale_orig'):
            feedback_module.neg_scale_orig = feedback_module.neg_scale
            feedback_module.neg_scale = nn.Parameter(feedback_module.neg_scale_orig * neg_scale_adjust)
            
        if pos_scale_adjust != 1.0 and not hasattr(feedback_module, 'pos_scale_orig'):
            feedback_module.pos_scale_orig = feedback_module.pos_scale
            feedback_module.pos_scale = nn.Parameter(feedback_module.pos_scale_orig * neg_scale_adjust)

    @torch.no_grad()
    # After applying steering, this method resets the modulation strengths to their default values.
    def reset_modulation_strengths(self):
        for _,lrm_module in self.lrm.named_children():            
            for _,feedback_module in lrm_module.named_children():
                if hasattr(feedback_module, 'neg_scale_orig'):
                    feedback_module.neg_scale = nn.Parameter(feedback_module.neg_scale_orig)
                    del feedback_module.neg_scale_orig
                if hasattr(feedback_module, 'pos_scale_orig'):
                    feedback_module.pos_scale = nn.Parameter(feedback_module.pos_scale_orig)
                    del feedback_module.pos_scale_orig
                    
    def forward(self, x, drop_state=True, forward_passes=None, return_list=None, 
                steering_signals=None, first_pass_steering=False):
        '''
            forward pass for Steerable LRMs
            
            Default behavior is to:
                - drop any stored activations
                - perform the default number of forward_passes (all modulated after 1st pass)
                - return the output of the final forward pass
            
            However, there are several possible overrides that can be enabled for steering.
            
            drop_state can be set to False to retain prior modulatory activation states,
            and the number of forward_passes can be specified. This could be useful
            if you wanted to manually step through N forwarded passes,
                out1 = model(x, drop_state=True, forward_passes=1)
                out2 = model(x, drop_state=False, forward_passes=1)
                out3 = model(x, drop_state=False, forward_passes=1)
                ...
            
            It's also possible to return the outputs of each forward pass (list_outputs=True).
            
            Finally, you can specify "cognitive steering" signals, using external (cognitive)
            representations to influence feed-forward processing (e.g., amplifying responses
            to features consistent with a stored memory representation of the prototypical "kit fox").         
            
            steering = [
                dict(source='classifier.6', template=target_prototype, strength=1, weight=1)
            ]
            
            source [str]: name of the module whose feedback activation you are overriding
            template [torch.tensor]:  steering activations, should be shape of source output inculding a batch dimension
            strength [float or _pair of floats]: multiplier_of_modulation_stength for neg/pos feedback 
                                                          (1=default modulation; 3 works well)
            weight: how much weight to give the steering_activation relative to the default 
                             activation 
                             
            What's the difference between steering `strength` and steering `weight`? 
            
            In LRMNetworks, negative and positive feedback is squashed by a tanh to be in the range
            [-1,1], and then we separately scale the negative and positive values by a learned neg_scale 
            and pos_scale parameter. When steering, we use neg_scale * steering_strength, pos_scale * steering_strength,
            allowing us to "boost" the feedback strength. It's essentially a global "effort" knob.
            
            steering_weight is different, and affects how much the feedback activation reflects the
            provided `template` vs. the original feedback activation (1=totally replace the default 
            activation with steering signal, 0=ignore steering signal).                         
                             
        '''
        return_list = self.return_list if return_list is None else return_list
        
        # Make sure steering_signals is a list
        if steering_signals is not None and isinstance(steering_signals, dict): 
            steering_signals = [steering_signals]
        
        # Optionally drop any stored feedback or skip inputs (from previous batch)
        if drop_state:
            for block in self.lrm.children(): 
                block.mod_inputs = {}
        
        # Optional steering for initial pass (e.g. "priming" or pre-trial attention)
        if steering_signals is not None and first_pass_steering:
            self.add_steering_signals(steering_signals)
        
        # iterate over forward_passes,
        # - input-drive is always the same (x)
        # - first pass not modulated by feedback (unless first_pass_steering)
        # - all subsequent passes modulated by feedback (feedback signal also updates on each pass)        
        outputs = []
        forward_passes = self.forward_passes if forward_passes is None else forward_passes
        for pass_num in range(0, forward_passes):
            out = self.feedforward(x)
            outputs.append(out)
            
            # After each pass, update native steering vector with provided steering vector(s)
            if steering_signals is not None:
                self.add_steering_signals(steering_signals)
        
        # reset feedback modulation strengths
        self.reset_modulation_strengths()
        
        if return_list:
            return outputs

        return out