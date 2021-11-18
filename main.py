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
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--crop_size', type=int, default=500)
parser.add_argument('--seed', type=int, default=28)
parser.add_argument('--rate', type=float, default=1e-4)
parser.add_argument('--image_dir', type=str, default='D:/div2k')
parser.add_argument('--save_dir', type=str, default='D:/div2k/')
parser.add_argument('--pretrain_path', type=str, default=None)
parser.add_argument('--optimizer', type=str, default='adam',choices=['adam', 'sgd', 'rmsprop'])
parser.add_argument('--loss', type=str, default='MSE', choices=['MSE', 'L1'])

arg = parser.parse_args()


def crop():
    return Compose([
        RandomCrop(arg.crop_size),
    ])

def transform():
    return Compose([
        ToTensor(),
    ])

def psnr(img1, img2):
    mse = torch.mean((img1-img2)**2)
    return 20*torch.log10((1.0/torch.sqrt(mse)))

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

def checkpoint(epoch):
    output_path = arg.save_dir+"checkpoint_cnn_epoch5.pth"
    torch.save(model.state_dict(), output_path)

if __name__ == '__main__':
    print('---Loading dataset---')
    train_dir = os.path.join(arg.image_dir,'train\\')
    train_set = DatasetFromFolder(train_dir, arg.upscale_fact, crop(), transform())
    train_data_loader = DataLoader(dataset=train_set,batch_size=arg.batch_size, shuffle=True)
    #show_sample(train_data_loader)

    pretrain = False
    model = CNN(channels=3, filters=64, features=256, scale_fact=arg.upscale_fact)

    if arg.pretrain_path is not None:
        pretrain = True
        model.load_state_dict(torch.load(arg.pretrain_path))

    if arg.loss == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.rate, betas=(0.9, 0.999), eps=1e-8)
    if not pretrain:
        for epoch in range(arg.num_epochs + 1):
            train(epoch)
            if epoch%10 == 0:
                checkpoint(epoch)

    # test output of model
    test_img = next(iter(train_data_loader))
    test_img_out = test_img[0][0]
    test_img_in = test_img[1][0].reshape(1, 3, 250, 250)
    test_output = model(test_img_in).detach()
    test_output = test_output.numpy().squeeze().transpose(1, 2, 0)
    plt.hist(test_output[:, :, 0].flatten(), color='red', alpha=0.5)
    plt.hist(test_output[:, :, 1].flatten(), color='green', alpha=0.5)
    plt.hist(test_output[:, :, 2].flatten(), color='blue', alpha=0.5)
    plt.show()
    plt.figure()
    plt.imshow(test_img[1][0].numpy().transpose(1,2,0))
    plt.figure()
    plt.imshow(test_output)
    plt.figure()
    plt.imshow(test_img_out.numpy().transpose(1,2,0))
    plt.show()
