import argparse
import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, CenterCrop

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import dataset
from dataset import DatasetFromFolder


# Parse settings for training
parser = argparse.ArgumentParser(description="Super Resolution Network Training")
parser.add_argument('--upscale_fact', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--crop_size', type=int, default=300)
parser.add_argument('--seed', type=int, default=28)
parser.add_argument('--rate', type=float, default=1e-4)
parser.add_argument('--image_dir', type=str, default='D:\div2k')
parser.add_argument('--optimizer', type=str, default='adam',choices=['adam','sgd','rmsprop'])

arg = parser.parse_args()


def crop():
    return Compose([
        CenterCrop(arg.crop_size),
    ])

def transform():
    return Compose([
        ToTensor(),
    ])

train_dir = os.path.join(arg.image_dir,'train\\')
train_set = DatasetFromFolder(train_dir, arg.upscale_fact, crop(), transform())
train_data_loader = DataLoader(dataset=train_set,batch_size=arg.batch_size, shuffle=True)

dataiter = iter(train_data_loader)
for i in range(1):
    images = dataiter.next()
    targets = images[0]
    inputs = images[1]
    bicubics = images[2]
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.imshow(np.transpose(targets[0].numpy(), (1, 2, 0)))
    ax2 = fig.add_subplot(132)
    ax2.imshow(np.transpose(inputs[0].numpy(), (1, 2, 0)))
    ax3 = fig.add_subplot(133)
    ax3.imshow(np.transpose(bicubics[0].numpy(), (1, 2, 0)))
    plt.show()
    print(inputs[0].numpy().shape)
    print(bicubics[0].numpy().shape)