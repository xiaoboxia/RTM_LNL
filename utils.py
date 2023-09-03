import os

import numpy as np
import torch
from option import args


def calculate_cauchy_parameter(my_loss, epoch):
    my_loss.calculate_epsilon(epoch)
    my_loss.calculate_gamma(epoch)


def get_dataset_size():
    if args.dataset == 'mnist':
        return 54000, 6000, 10000
