import argparse
import os.path
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, RandomCrop, Resize

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from dataset import DatasetFromFolder
from model import Net as CNN
from dbpn import Net as DBPN

# Parse settings for training
parser = argparse.ArgumentParser(description="Super Resolution Network Training")
parser.add_argument('--model_type', type=str, default='CNN', choices=['CNN', 'DBPN'])
parser.add_argument('--upscale_fact', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--crop_size', type=int, default=100)
parser.add_argument('--seed', type=int, default=539)
parser.add_argument('--rate', type=float, default=1e-4)
parser.add_argument('--image_dir', type=str, default='D:\div2k')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
parser.add_argument('--loss', type=str, default='L1', choices=['L1', 'L2', 'MSE', 'Sparse'])
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--cuda', type=bool, default=False)

arg = parser.parse_args()


def crop():
    return Compose([
        RandomCrop(arg.crop_size),
    ])


def resize():
    return Compose([
        Resize((arg.crop_size, arg.crop_size)),
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
    ax1.title.set_text("Target\n" + str(tar_shape))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(targets)
    ax2 = fig.add_subplot(131)
    ax2.title.set_text("Input\n" + str(in_shape))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(inputs)
    ax3 = fig.add_subplot(132)
    ax3.title.set_text("Bicubic\n" + str(tar_shape))
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(bicubics)
    plt.show()

def show_output(data_loader, model):
    data_iter = iter(data_loader)
    images = data_iter.next()
    inputs = np.transpose(images[1][0].numpy(), (1, 2, 0))
    targets = np.transpose(images[0][0].numpy(), (1, 2, 0))
    bicubics = np.transpose(images[2][0].numpy(), (1, 2, 0))
    pred = model(images[1])
    pred = pred[0].detach().numpy()
    pred = np.transpose(pred,(1,2,0))
    print(pred.shape)
    fig = plt.figure()
    ax1 = fig.add_subplot(141)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.title.set_text('Input')
    ax1.imshow(inputs)
    ax2 = fig.add_subplot(142)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.title.set_text('Bicubic')
    ax2.imshow(bicubics)
    ax3 = fig.add_subplot(143)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.title.set_text('Prediction')
    ax3.imshow(pred)
    ax4 = fig.add_subplot(144)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.title.set_text('Ground Truth')
    ax4.imshow(targets)
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
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}.".format(epoch, iteration, len(train_data_loader), loss.data))
    return curr_loss


def test():
    avg_psnr = 0
    for batch in val_data_loader:
        target, image_in = Variable(batch[0]), Variable(batch[1])

        prediction = model(image_in)
        mse = nn.functional.mse_loss(prediction, target)
        psnr = 10 * math.log10(1 / mse.data)
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(val_data_loader)))


def checkpoint(iterate):
    model_out_path = './weights/' + arg.model_type + "_" + arg.loss + "_epoch_{}.pth".format(iterate)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def evaluate_history(history):

    num_epochs = len(history)
    unit = num_epochs / 5

    # 学習曲線の表示 (損失): learning curve (loss)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='Training')
    #plt.plot(history[:,0], history[:,3], 'k', label='Test(Val)')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('# of recursion')
    plt.ylabel('Loss')
    plt.title('Learning Curve (Loss)')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    print('---Loading dataset---')
    train_dir = os.path.join(arg.image_dir, 'train\\')
    val_dir = os.path.join(arg.image_dir, 'val\\')
    train_set = DatasetFromFolder(train_dir, arg.upscale_fact, resize(), transform())
    train_data_loader = DataLoader(dataset=train_set, batch_size=arg.batch_size, shuffle=True)
    val_set = DatasetFromFolder(val_dir, arg.upscale_fact, resize(), transform())
    val_data_loader = DataLoader(dataset=val_set, batch_size=arg.batch_size, shuffle=True)
    # show_sample(train_data_loader)

    # select model type
    if arg.model_type == 'CNN':
        model = CNN(channels=3, filters=64, features=256, scale_fact=arg.upscale_fact)
    elif arg.model_type == 'DBPN':
        model = DBPN(num_channels=3, base_filter=64, feat=256, num_stages=7, scale_factor=arg.upscale_fact)

    # select loss type
    if arg.loss == 'MSE' or arg.loss == 'L2':
        criterion = nn.MSELoss()
    elif arg.loss == 'Sparse' or arg.loss == 'L1':
        criterion = nn.L1Loss()

    # select optimizer type
    if arg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.rate, betas=(0.9, 0.999), eps=1e-8)
    elif arg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=arg.rate)
    elif arg.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=arg.rate)

    if arg.pretrained:
        new_state_dict = torch.load('./weights/'+arg.model_type+'_'+arg.loss+'_epoch_20.pth', map_location=torch.device('cpu'))
        model.load_state_dict(new_state_dict)
        print('---Pretrained model is loaded---')

    # check if training uses cuda
    if arg.cuda:
        # add gpu support
        print("need to add gpu support")

    if not arg.pretrained:
        history = np.zeros((0, 2))
        for epoch in range(1, arg.num_epochs + 1):
            loss = train(epoch)

            # learning rate is decayed by a factor of 10 every half of total epochs
            if epoch % (arg.num_epochs / 2) == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10  # 10^(1/2)
                print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

            if epoch % 10 == 0:
                checkpoint(epoch)
            item = np.array([epoch + 1, loss])
            history = np.vstack((history, item))
        evaluate_history(history)

    # test the current model
    test()
    show_output(val_data_loader, model)

