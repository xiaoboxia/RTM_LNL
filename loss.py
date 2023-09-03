from typing import Any
from torch import nn
import torch
from torch.distributions.normal import Normal
import numpy as np
from option import args
inner_ce = nn.CrossEntropyLoss(reduction='none')


class PManager:
    epsilon = 1
    alpha = 1

    @classmethod
    def reset(cls):
        cls.epsilon = 1
        cls.alpha = 1

    @classmethod
    def epsilon_estimate(cls, target, ce):
        if args.parafix:
            return 1
        if args.ablation_fix != 0:
            return args.ablation_fix

        ce = ce.detach().sort().values
        best_diff = np.inf
        for i in np.arange(0.8, args.range+0.1, 0.1):
            lp_loss = torch.log(1 + ce / i)
            diff = (target - lp_loss).abs().sum()
            if best_diff > diff:
                cls.epsilon = i
                best_diff = diff
        
        return cls.epsilon

    @classmethod
    def alpha_estimate(cls, target, ce):
        if args.parafix:
            return 1
        if args.ablation_fix != 0:
            return args.ablation_fix
        ce = ce.detach().sort().values
        best_diff = np.inf
        for i in np.arange(0.8, args.range+0.1, 0.1):
            wp_loss = 1 - np.e **(- ce / i)
            diff = (target - wp_loss).abs().sum()
            if best_diff > diff:
                cls.alpha = i
                best_diff = diff

        return cls.alpha

    @classmethod
    def cop_estimate(cls, ce_loss):
        # estimate cut of point, i.e., threshold with three-sigma-rule
        med = torch.median(ce_loss).detach()
        Tau = ce_loss[ce_loss <= med].detach()
        mean = torch.mean(Tau)
        std = torch.std(Tau)
        epsilon = 3 * std + mean
        return epsilon
    
    @classmethod
    def two_cop_estimate(cls, ce_loss):
        # estimate cut of point, i.e., threshold with three-sigma-rule
        med = torch.median(ce_loss).detach()
        Tau = ce_loss[ce_loss <= med].detach()
        mean = torch.mean(Tau)
        std = torch.std(Tau)
        epsilon = 3 * std + mean
        epsilon_left = mean - 3*std
        return epsilon, epsilon_left


    @classmethod
    def generate_normal(cls, loss):
        u = loss.mean().detach()
        std = loss.std().detach()
        normal_sample = Normal(u, std).sample([loss.shape[0]])
        normal_sample.sort().values
        return normal_sample


def catoni(yhat, y, *useless):
    ce_loss = inner_ce(yhat, y)
    return torch.log(1 + ce_loss + ce_loss**2).mean()


def tcatoni(yhat, y, epoch):
    ce_loss = inner_ce(yhat, y)
    cop = PManager.cop_estimate(ce_loss)
    selected_ce_loss = ce_loss[(ce_loss <= cop)]
    if args.threshold_offset != 0:
        memorize_rate = (ce_loss <= cop).float().detach().cpu().numpy().mean()
        memorize_rate += args.threshold_offset / 100
        c = ce_loss.detach().cpu().numpy()
        percentile = min(max(memorize_rate * 100, 0), 100)
        if percentile == 0 or percentile == 100:
            print('WTF??????????????')
        new_cop = np.percentile(c, percentile)
        # print(f"epsilon difference {(ce_loss <= cop).float().detach().cpu().numpy().mean() - memorize_rate}")
        selected_ce_loss = ce_loss[(ce_loss <= new_cop)]

    return torch.log(1 + selected_ce_loss + selected_ce_loss**2).mean()


def rtcatoni(yhat, y, epoch):
    if epoch % args.relax == 0:
        return catoni(yhat, y)
    else:
        return tcatoni(yhat, y, epoch)


def logsum(yhat, y, *useless):
    ce_loss = inner_ce(yhat, y)
    lp_loss = torch.log(1 + ce_loss / PManager.epsilon)
    normal_sample = PManager.generate_normal(lp_loss)
    epsilon = PManager.epsilon_estimate(normal_sample, ce_loss)
    # epsilon = 1
    return torch.log(1 + ce_loss / epsilon).mean()


def tlogsum(yhat, y, epoch):
    ce_loss = inner_ce(yhat, y)
    cop = PManager.cop_estimate(ce_loss)
    tlp_loss = torch.log(1 + ce_loss[ce_loss <= cop] / PManager.epsilon)
    normal_sample = PManager.generate_normal(tlp_loss)
    epsilon = PManager.epsilon_estimate(normal_sample, ce_loss[ce_loss <= cop])
    # epsilon = 1
    if args.threshold_offset != 0:
        memorize_rate = (ce_loss <= cop).float().detach().cpu().numpy().mean()
        memorize_rate += args.threshold_offset / 100
        c = ce_loss.detach().cpu().numpy()
        percentile = min(max(memorize_rate * 100, 0), 100)
        if percentile == 0 or percentile == 100:
            print('WTF??????????????')
        new_cop = np.percentile(c, percentile)
        # print(f"epsilon difference {(ce_loss <= cop).float().detach().cpu().numpy().mean() - memorize_rate}")
        cop = new_cop

    return torch.log(1 + ce_loss[ce_loss <= cop] / epsilon).mean()


def rtlogsum(yhat, y, epoch):
    if epoch % args.relax == 0:
        return logsum(yhat, y)
    else:
        return tlogsum(yhat, y, epoch)


def welschp(yhat, y, *useless):
    ce_loss = inner_ce(yhat, y)
    wp_loss = 1 - np.e **(- ce_loss / PManager.alpha)
    normal_sample = PManager.generate_normal(wp_loss)
    alpha = PManager.alpha_estimate(normal_sample, ce_loss)
    # alpha = 1
    return (1 - np.e **(- ce_loss / alpha)).mean()

def twelschp(yhat, y, epoch):       
    ce_loss = inner_ce(yhat, y)
    cop = PManager.cop_estimate(ce_loss)
    twp_loss = (1 - np.e **(- ce_loss[(ce_loss <= cop)] / PManager.alpha))
    normal_sample = PManager.generate_normal(twp_loss)
    alpha = PManager.alpha_estimate(normal_sample, ce_loss[ce_loss <= cop])
    # alpha = 1
    if args.threshold_offset != 0:
        memorize_rate = (ce_loss <= cop).float().detach().cpu().numpy().mean()
        memorize_rate += args.threshold_offset / 100
        c = ce_loss.detach().cpu().numpy()
        percentile = min(max(memorize_rate * 100, 0), 100)
        if percentile == 0 or percentile == 100:
            print('WTF??????????????')
        new_cop = np.percentile(c, percentile)
        # print(f"epsilon difference {(ce_loss <= cop).float().detach().cpu().numpy().mean() - memorize_rate}")
        cop = new_cop



    return (1 - np.e **(- ce_loss[ce_loss <= cop] / alpha)).mean()

def rtwelschp(yhat, y, epoch):
    if epoch % args.relax == 0:
        return welschp(yhat, y)
    else:
        return twelschp(yhat, y, epoch)


class MyCE():
    def __init__(self) -> None:
        self.ce = nn.CrossEntropyLoss()
    
    def __call__(self, a, b, epoch):
        return self.ce(a,b)


def select_loss():
    if args.loss == 'ce':
        return MyCE()
    elif args.loss == 'catoni':
        return catoni
    elif args.loss == 'logsum':
        return logsum
    elif args.loss == 'welschp':
        return welschp
    elif args.loss == 'tcatoni':
        return tcatoni
    elif args.loss == 'tlogsum':
        return tlogsum
    elif args.loss == 'twelschp':
        return twelschp
    elif args.loss == 'rtcatoni':
        return rtcatoni
    elif args.loss == 'rtlogsum':
        return rtlogsum
    elif args.loss == 'rtwelschp':
        return rtwelschp
