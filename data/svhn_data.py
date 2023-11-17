from PIL import Image
import os
import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
from torch.utils.data import DataLoader
import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from data.make_label_noise import  v6_get_noisy_label, noisify
from data.transform import transform_test, transform_train
from option import args
from PIL import Image

class SVHN(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(SVHN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index: int):

        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.to_tensor(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)




def get_svhn_data(flip_percentage):
    # Currently, i don't transform
    os.makedirs(args.data_path, exist_ok=True)
    test_transform = transform_test(args.dataset)
    train_transform = transform_train(args.dataset)

    test_data = SVHN(args.data_path, split='test', transform=test_transform, download=True)
    raw_train_data = SVHN(args.data_path, split='train', transform=train_transform, download=True)

    test_data_loader = DataLoader(dataset=test_data, shuffle=False, pin_memory=False,
                                    batch_size=args.batch_size, num_workers=args.n_threads)

    if args.noise_type == 'ILN':
        train_data_for_noise = SVHN(args.data_path, split='train',
                                                            transform=transforms.ToTensor(), download=True)
        new_targets = v6_get_noisy_label(flip_percentage, train_data_for_noise, train_data_for_noise.labels)
    elif args.noise_type == 'pairflip' or args.noise_type == 'symmetric':
        new_targets, _ = noisify(10, raw_train_data.labels)

    is_noise = (new_targets != torch.LongTensor(raw_train_data.labels)).float().to(args.device)
    raw_train_data.labels = new_targets
    print(f'dataset training set size  = {len(raw_train_data)}')
    train_data, val_data = torch.utils.data.random_split(raw_train_data, [65932, 7325])
    train_data_loader = DataLoader(dataset=train_data, shuffle=True, pin_memory=False,
                                    batch_size=args.batch_size, num_workers=args.n_threads)
    val_data_loader = DataLoader(dataset=val_data, shuffle=False, pin_memory=False,
                                    batch_size=args.batch_size, num_workers=args.n_threads)

    return raw_train_data, train_data_loader, val_data_loader, test_data_loader, is_noise
