import os
import pickle
import sys

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset, CIFAR10
from data.make_label_noise import v6_get_noisy_label, noisify
from data.transform import transform_test, transform_train
from option import args
from PIL import Image


class News(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(News, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.train = train  # training set or test set
        self.to_tensor = torchvision.transforms.ToTensor()
        if train:
            self.data = np.load(f'{root}train_data.npy')
            self.targets = np.load(f'{root}train_label.npy')
        else:
            self.data = np.load(f'{root}test_data.npy')
            self.targets = np.load(f'{root}test_label.npy')

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img, mode='L')
        img = torch.from_numpy(img)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        # if self.transform is not None:
        #     img = self.transform(img)
        # else:
        #     img = self.to_tensor(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)
def get_news_data(flip_percentage):
    # Currently, i don't transform
    os.makedirs(args.data_path, exist_ok=True)
    test_transform = transform_test(args.dataset)
    train_transform = transform_train(args.dataset)

    test_data = News(args.data_path, train=False, transform=test_transform, download=True)
    raw_train_data = News(args.data_path, train=True, transform=train_transform, download=True)

    test_data_loader = DataLoader(dataset=test_data, shuffle=False, pin_memory=False,
                                  batch_size=args.batch_size, num_workers=args.n_threads)

    if args.noise_type == 'ILN':
        train_data_for_noise = News(args.data_path, train=True,
                                    transform=transforms.ToTensor(), download=True)
        new_targets = v6_get_noisy_label(flip_percentage, train_data_for_noise, train_data_for_noise.targets)
    elif args.noise_type == 'pairflip' or args.noise_type == 'symmetric':
        new_targets, _ = noisify(20, raw_train_data.targets)

    is_noise = (new_targets != torch.LongTensor(raw_train_data.targets)).float().to(args.device)
    raw_train_data.targets = new_targets
    train_data, val_data = torch.utils.data.random_split(raw_train_data, [10183, 1131])
    train_data_loader = DataLoader(dataset=train_data, shuffle=True, pin_memory=False,
                                   batch_size=args.batch_size, num_workers=args.n_threads)
    val_data_loader = DataLoader(dataset=val_data, shuffle=False, pin_memory=False,
                                 batch_size=args.batch_size, num_workers=args.n_threads)

    return raw_train_data, train_data_loader, val_data_loader, test_data_loader, is_noise
