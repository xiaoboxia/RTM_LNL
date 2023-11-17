import os

import numpy as np
import torch
from option import args


def calculate_cauchy_parameter(my_loss, epoch):
    my_loss.calculate_epsilon(epoch)
    my_loss.calculate_gamma(epoch)


def get_dataset_size():
    if args.dataset == 'cifar10' or args.dataset == 'cifar100'  or args.dataset == 'cifar10n':
        return 45000, 5000, 10000
    elif args.dataset == 'svhn':
        return 65932, 7325, 26032
    elif args.dataset == 'mnist':
        return 54000, 6000, 10000
    elif args.dataset == 'news':
        return 10183, 1131, 7532
        