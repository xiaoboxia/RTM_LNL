import argparse
# import template
import datetime
import sys
import torch

parser = argparse.ArgumentParser(description='label noise')

parser.add_argument('--n_threads', type=int, default=1,
                    help='number of threads for data loading')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--device', default=0, type=int,
                    help='0|1 for different gpu')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='mnist|cifar10|cifar100; mnist only for FC or simple network')
parser.add_argument('--model', default='FC',
                    help='model name: simple_network|FC|resnet18|resnet32|resnet50')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--repeat', type=int, default=5,
                    help='experiment repeating time for each flip')
parser.add_argument('--relax', type=int, default=2,
                    help='X: relax epsilon every x epochs. -1 for no relax')
parser.add_argument('--noise_rate', type=float, default=50)
parser.add_argument('--start_time', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d %H:%M-%S'))
parser.add_argument('--data_path', type=str)
parser.add_argument('--noise_type', type=str, default='ILN', help='symmetric, pairflip, ILN')
parser.add_argument('--use_aug', type=str, default='True')
parser.add_argument('--loss', type=str, default='rtcatoni', help='ce|catoni|logsum|welschp|tcatoni|tlogsum|twelschp|rtcatoni|rtwelschp|rtlogsum')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=0.001)
parser.add_argument('--sleep', type=float, default=0)
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--two_cop', type=str, default='False')
parser.add_argument('--a', type=int, default=40)
parser.add_argument('--b', type=int, default=80)
parser.add_argument('--parafix', type=str, default='False')
parser.add_argument('--threshold_offset', type=int, default=0)
parser.add_argument('--ablation_fix', type=float, default=0)
parser.add_argument('--save_file', type=str, default='results')
parser.add_argument('--pretrain_lr', type=float, default=0.01)
parser.add_argument('--pretrain_wd', type=float, default=0.001)


args = parser.parse_args()

import sys
print('python', ' '.join(sys.argv))
args.device = torch.device(f"cuda:{args.device}")


print(args.device)
print('****************')
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

if args.dataset == 'mnist':
    args.model = 'FC'
    args.data_path = "./database/mnist"
elif args.dataset == 'cifar10':
    args.model = 'resnet18'
    args.data_path = "./database/cifar10"
elif args.dataset == 'svhn':
    args.model = 'resnet18'
    args.data_path = "./database/svhn/"
elif args.dataset == 'news':
    args.model = 'newsnet'
    args.data_path = "./database/news/"
elif args.dataset == 'cifar100':
    args.model = 'resnet50'
    args.data_path = "./database/cifar100"


import json

file_path = f"./config/{args.dataset}.json"

import os 
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
        for item in config['configurations']:
            if item['loss'] == args.loss and item['noise_rate'] == args.noise_rate and item['noise_type'] == args.noise_type:
                if 'pretrain' in item['settings']:
                    args.pretrain = item['settings']['pretrain']
                if 'relax' in item['settings']:
                    args.relax = item['settings']['relax']
                break
