import numpy as np
from PIL import Image
import cv2


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


def upscale_img(img_path, scale):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(img_path[:-4] + "_scaled.png", img)


if __name__ == '__main__':
    upscale_img("/home/zartris/Pictures/BHS/BHS_Testing_presentation.png", 10)
