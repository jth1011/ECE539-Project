from PIL import Image, ImageOps
import torch
import torch.utils.data as data
import numpy as np
from os.path import join
from os import listdir


def is_img_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filename):
    return Image.open(filename).convert('RGB')


def rescale_img(img, scale):
    size_in = img.size
    size_out = tuple([int(x * scale) for x in size_in])
    return img.resize(size_out, resample=Image.BICUBIC)


class DatasetFromFolder(data.Dataset):
    def __init__(self, img_dir, upscale_factor, crop=None, transform=None):
        self.img_files = [join(img_dir, x) for x in listdir(img_dir) if is_img_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.crop = crop

    def __getitem__(self, item):
        target = load_img(self.img_files[item])

        if self.crop:
            target = self.crop(target)

        img_in = target.resize((int(target.size[0] / self.upscale_factor), int(target.size[1] / self.upscale_factor)),
                               Image.BICUBIC)
        bicubic = rescale_img(img_in, self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            img_in = self.transform(img_in)
            bicubic = self.transform(bicubic)

        return target, img_in, bicubic

    def __len__(self):
        return len(self.img_files)
