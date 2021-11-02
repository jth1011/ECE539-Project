import argparse

import torch
import torch.nn as nn
from torch.utils.data import dataloader

import numpy as np


# Parse settings for training
parser = argparse.ArgumentParser(description="Super Resolution Network Training")
parser.add_argument('--upscale_fact', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--seed', type=int, default=28)
parser.add_argument('--rate', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='Adam',choices=['Adam','SGD','rmsprop'])
parser.add_argument('--regulizer', type=str, default='None', choices=['None','L1','L2'])

arg = parser.parse_args()