
import os
from pprint import pprint
import torch

cpu_num = 1 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

from option import args
import time
from loss import PManager, select_loss
import sys
from math import inf

import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from data.data_choose import get_data
from networkbk.choose_network import network_choose
from utils import  get_dataset_size


flip = args.noise_rate
print(f'flip={flip}')
print(f'pid = {os.getpid()}')
best = []
os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID' 


pprint(vars(args))
time.sleep(args.sleep)

time1 = time.time()

for repeat in range(args.repeat):
    args.seed = repeat + 1
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    raw_train_data, train_data_loader, val_data_loader, test_data_loader, is_noise = get_data(flip / 100)
    train_data_size, val_data_size, test_data_size = get_dataset_size()

    net = network_choose()  # type: nn.Module


    os.makedirs(f'./model/{args.dataset}', exist_ok=True)
    if args.pretrain > 0:
        ce = nn.CrossEntropyLoss()
        path = f'./model/{args.dataset}/pre_{args.lr}_{args.wd}_{args.a}{args.b}_{args.pretrain}_{repeat}.pth.tar'
        optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
        scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
        from tqdm import trange
        for epoch in trange(args.pretrain):
            net.train()
            for x, y, idx in train_data_loader:
                x, y = x.to(args.device), y.to(args.device)
                y_hat = net(x)
                loss_value = ce(y_hat, y)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
        scheduler.step()
        state = {
        'state_dict': net.state_dict(),
        }
        # torch.save(state, path)
        
    
    optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.wd)
    scheduler = MultiStepLR(optimizer, milestones=[args.a, args.b], gamma=0.1)
    loss = select_loss()


    best_epoch = 0
    best_val_acc, best_test_acc = 0, 0
    PManager.reset()


    for epoch in tqdm(range(args.epochs), position=1, file=sys.stdout):
        print(flush=True)
        print(f'current seed = {repeat}')

        train_loss, train_loss2 = 0, 0
        train_correct, train_correct2 = 0, 0

        net.train()
        for x, y, idx in train_data_loader:
            x, y = x.to(args.device), y.to(args.device)
            y_hat = net(x)
            loss_value = loss(y_hat, y, epoch)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            _, pred = torch.max(y_hat, 1)
            train_correct += torch.sum(pred == y.data).item()

        scheduler.step()

        val_correct, val_correct2 = 0, 0
        with torch.no_grad():
            net.eval()
            for x, y, idx in val_data_loader:
                x, y = x.to(args.device), y.to(args.device)
                y_hat = net(x)
                _, pred = torch.max(y_hat, 1)
                val_correct += torch.sum(pred == y.data).item()

        testing_correct, testing_correct2 = 0, 0
        with torch.no_grad():
            net.eval()
            for x, y, idx in test_data_loader:
                x, y = x.to(args.device), y.to(args.device)
                y_hat = net(x)
                _, pred = torch.max(y_hat, 1)
                testing_correct += torch.sum(pred == y.data).item()

        if best_val_acc < val_correct / val_data_size:
            best_val_acc = val_correct / val_data_size
            old_best_test_acc = best_test_acc
            best_test_acc = testing_correct / test_data_size
            best_epoch = epoch
            print(f'best_test_changed = {best_test_acc - old_best_test_acc:.4f}')

        print(f"train_acc = {train_correct / train_data_size:.4f}")

        print(f'current best acc of model1 = {best_test_acc:.4f}')
        print(f"current best val of model1 = {best_val_acc:.4f}")
        print(f'current best epoch of model1 = {best_epoch}')
        print(f"test_acc = {testing_correct / test_data_size: .4f}, "
                f"eval_acc = {val_correct / val_data_size : .4f}", flush=True)
        print(f'{best}')
        
    best.append(best_test_acc)


result = np.array(best)
meanstd = f'{result.mean()*100:.2f}+-{result.std()*100:.2f}'
print(meanstd)

time2 = time.time()
print('time=', time2 - time1)

with open(f'./results/{args.save_file}.csv','a') as f:
    f.write(f'{args.dataset},{args.noise_type},{args.noise_rate},{args.loss},{meanstd},{args.pretrain},{args.relax},{args.ablation_fix},{result.mean()*100:.2f},{args.two_cop},{args.lr},{args.wd},{args.a},{args.b},{args.threshold_offset},{args.use_aug},{args.pretrain_lr},{args.pretrain_wd}\n')

