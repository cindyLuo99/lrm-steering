from .download import get_remote_data_file
from .folder import ImageFolderIndex

__all__ = ['imagenette2']

def imagenette2(split, transform=None):
  url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
  cached_filename, extracted_folder = get_remote_data_file(url)
  dataset = ImageFolderIndex(os.path.join(extracted_folder, split), transform=transform)
  return dataset

  