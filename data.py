from PIL import Image, ImageOps
import torch
import torch.utils.data as data
import numpy as np
from os.path import join


def is_img_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filename):
    return Image.open(filename).convert('RGB')

def rescale_img(img, scale):
    size_in = img.size
    size_out = tuple([int(x * scale) for x in size_in])
    return img.resize(size_out,resample=Image.BICUBIC)

def get_train_data(data_dir, upscale_factor):
    dir = join(data_dir, 'train/')
    return DatasetFromFolder(dir,upscale_factor)

def get_val_data(data_dir, upscale_factor):
    dir = join(data_dir, 'val/')
    return DatasetFromFolder(dir, upscale_factor)