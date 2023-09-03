import numpy as np
import os
from hashlib import sha1

import torch
from scipy import stats
from scipy.special import softmax

from option import args

norm_std = 0.1 

def v6_get_noisy_label(n, dataset, labels):
    if args.dataset == 'cifar100':
        label_num = 100
    elif args.dataset == 'news':
        label_num = 20
    else:
        label_num = 10
    os.makedirs(f'{args.data_path}', exist_ok=True)
    file_path = f"{args.data_path}/v6_{args.dataset}_labels_{n}_{args.seed}.npy"
    print(file_path)
    if os.path.exists(file_path):
        new_label = np.load(file_path)
    else:
        from numpy.random import randn as rn
        from math import inf
        from torch import nn
        P = []
        if args.dataset == 'mnist':
            feature_size = 28 * 28
        elif args.dataset.startswith('cifar'):
            feature_size = 3 * 32 * 32
        elif args.dataset.startswith('svhn'):
            feature_size = 3 * 32 * 32
        elif args.dataset.startswith('news'):
            feature_size = 300

        flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n,
                                            scale=norm_std)
        flip_rate = flip_distribution.rvs(dataset.data.shape[0])

        if not isinstance(labels, torch.FloatTensor):
            try:
                labels = torch.FloatTensor(labels)
            except TypeError:
                labels = labels.float()
                
        labels = labels.to(args.device)

        W = np.random.randn(label_num, feature_size, label_num)
        W = torch.FloatTensor(W).to(args.device)
        for i, (x, y, idx) in enumerate(dataset):
            # 1*m *  m*10 = 1*10
            x = x.to(args.device)
            # print(x.type(), end='\n\n\n')
            A = x.view(1, -1).mm(W[y]).squeeze(0)
            A[y] = -inf
            A = flip_rate[i] * torch.softmax(A, dim=0)
            A[y] += 1 - flip_rate[i]
            P.append(A)
        P = torch.stack(P, 0).cpu().numpy()
        l = [i for i in range(label_num)]
        new_label = [np.random.choice(l, p=P[i]) for i in range(len(dataset))]

        np.save(file_path, np.array(new_label))
        print(f'noise rate = {(new_label != np.array(labels.cpu())).mean()}')

        record = [[0 for _ in range(label_num)] for i in range(label_num)]

        for a, b in zip(labels, new_label):
            a, b = int(a), int(b)
            record[a][b] += 1
        #
        print('****************************************')
        print('following is flip percentage:')

        for i in range(label_num):
            sum_i = sum(record[i])
            for j in range(label_num):
                if i != j:
                    print(f"{record[i][j] / sum_i: .2f}", end='\t')
                else:
                    print(f"{record[i][j] / sum_i: .2f}", end='\t')
            print()

        pidx = np.random.choice(range(P.shape[0]), 1000)
        cnt = 0
        for i in range(1000):
            if labels[pidx[i]] == 0:
                a = P[pidx[i], :]
                for j in range(label_num):
                    print(f"{a[j]:.2f}", end="\t")
                print()
                cnt += 1
            if cnt >= 10:
                break
    return torch.LongTensor(new_label)


# basic function
def multiclass_noisify(y, P, random_state):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    y = y.cpu().numpy()
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    from numpy.testing import assert_array_almost_equal
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)
    l = [i for i in range(P.shape[0])]
    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        # flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        # new_y[idx] = np.where(flipped == 1)[0]
        new_y[idx] = np.random.choice(l, p=P[i])
    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise
    np.random.seed(args.seed)

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        y_train_noisy = torch.from_numpy(y_train_noisy)
        actual_noise = (y_train_noisy != y_train).float().mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P
    np.random.seed(args.seed)
    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        y_train_noisy = torch.from_numpy(y_train_noisy)
        actual_noise = (y_train_noisy != y_train).float().mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def noisify(nb_classes=10, train_labels=None):
    noise_rate = args.noise_rate / 100
    print(noise_rate)
    train_labels = torch.LongTensor(train_labels)
    if args.noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate,
                                                                 nb_classes=nb_classes)
    if args.noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate,
                                                                             nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate
