import argparse
import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, RandomCrop

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from dataset import DatasetFromFolder
from model import Net as CNN


# Parse settings for training
parser = argparse.ArgumentParser(description="Super Resolution Network Training")
parser.add_argument('--upscale_fact', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--crop_size', type=int, default=500)
parser.add_argument('--seed', type=int, default=28)
parser.add_argument('--rate', type=float, default=1e-4)
parser.add_argument('--image_dir', type=str, default='D:\div2k')
parser.add_argument('--optimizer', type=str, default='adam',choices=['adam','sgd','rmsprop'])

arg = parser.parse_args()


def crop():
    return Compose([
        RandomCrop(arg.crop_size),
    ])

def transform():
    return Compose([
        ToTensor(),
    ])

def show_sample(data_loader):
    data_iter = iter(data_loader)
    images = data_iter.next()
    targets = np.transpose(images[0][0].numpy(), (1, 2, 0))
    inputs = np.transpose(images[1][0].numpy(), (1, 2, 0))
    bicubics = np.transpose(images[2][0].numpy(), (1, 2, 0))
    tar_shape = targets.shape
    in_shape = inputs.shape
    fig = plt.figure()
    ax1 = fig.add_subplot(133)
    ax1.title.set_text("Target\n"+str(tar_shape))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(targets)
    ax2 = fig.add_subplot(131)
    ax2.title.set_text("Input\n"+str(in_shape))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(inputs)
    ax3 = fig.add_subplot(132)
    ax3.title.set_text("Bicubic\n"+str(tar_shape))
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(bicubics)
    plt.show()

def train(epoch):
    curr_loss = 0
    model.train()
    for iteration, batch in enumerate(train_data_loader, 1):
        target, image_in, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        optimizer.zero_grad()
        pred = model(image_in)
        loss = criterion(pred, target)
        curr_loss += loss.data
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}.".format(epoch, iteration, len(train_data_loader), loss.data))


if __name__ == '__main__':
    print('---Loading dataset---')
    train_dir = os.path.join(arg.image_dir,'train\\')
    train_set = DatasetFromFolder(train_dir, arg.upscale_fact, crop(), transform())
    train_data_loader = DataLoader(dataset=train_set,batch_size=arg.batch_size, shuffle=True)
    #show_sample(train_data_loader)

    model = CNN(channels=3, filters=64, features=256, scale_fact=arg.upscale_fact)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.rate)
    train(0)