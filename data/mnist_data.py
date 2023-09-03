import os
from typing import List
import warnings

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.mnist import read_label_file, read_image_file
from torchvision.datasets.utils import  download_and_extract_archive
from os import makedirs
from data.make_label_noise import v6_get_noisy_label, noisify
from data.transform import transform_test, transform_train
from option import args


def makedir_exist_ok(path):
    makedirs(path, exist_ok=True)

class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.train = train  # training set or test set
        self.do_data_augmentation = set()

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.real_label = self.targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


def get_mnist_data(flip_percentage):
    os.makedirs(args.data_path, exist_ok=True)
    test_transform = transform_test(args.dataset)
    train_transform = transform_train(args.dataset)
    test_data = MNIST(args.data_path, train=False, transform=test_transform, download=True)
    raw_train_data = MNIST(args.data_path, train=True, transform=train_transform, download=True)
    test_data_loader = DataLoader(dataset=test_data, shuffle=False,
                                  batch_size=args.batch_size, num_workers=args.n_threads)

    if args.noise_type == 'ILN':
        train_data_for_noise = MNIST(args.data_path, train=True,
                                                          transform=transforms.ToTensor(),
                                                          download=True)
        new_targets = v6_get_noisy_label(flip_percentage, train_data_for_noise, train_data_for_noise.targets)

    elif args.noise_type == 'pairflip' or args.noise_type == 'symmetric':
        new_targets, _ = noisify(10, raw_train_data.targets)

    is_noise = (new_targets != torch.LongTensor(raw_train_data.targets)).float().to(args.device)
    raw_train_data.targets = new_targets
    train_data, val_data = torch.utils.data.random_split(raw_train_data, [54000, 6000])
    train_data_loader = DataLoader(dataset=train_data, shuffle=True,
                                   batch_size=args.batch_size, num_workers=args.n_threads)
    val_data_loader = DataLoader(dataset=val_data, shuffle=False,
                                 batch_size=args.batch_size, num_workers=args.n_threads)

    return raw_train_data, train_data_loader, val_data_loader, test_data_loader, is_noise
