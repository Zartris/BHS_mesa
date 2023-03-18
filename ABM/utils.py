import numpy as np
from PIL import Image


def load_image_to_np(path_to_map_file, convert=None, remove_alpha=True, rotate=None):
    if convert is not None:
        img = Image.open(path_to_map_file).convert(convert)
    else:
        img = Image.open(path_to_map_file)
    img = np.array(img)
    if remove_alpha:
        img = img[:, :, :3]
    if rotate is not None:
        img = np.rot90(img, rotate)
    return img
