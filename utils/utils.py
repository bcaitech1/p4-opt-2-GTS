import torch
import torch.nn as nn 
import torch.nn.functional as F
from ptflops import get_model_complexity_info

import sys
import os
import time


def calc_macs(model, input_shape):
    macs, _ = get_model_complexity_info(
        model=model,
        input_res=input_shape,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        ignore_modules=[nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6],
    )
    return macs

def loss_fn(outputs, targets, device):
    if len(targets.shape) == 1:
        return F.cross_entropy(outputs, targets)
    else:
        return torch.mean(torch.sum(-targets * F.log_softmax(outputs, dim=1), dim=1))

def label_smooth_loss_fn(outputs, targets, device, epsilon=0.1):
    onehot = F.one_hot(targets, 9).float().to(device)
    targets = (1 - epsilon) * onehot + torch.ones(onehot.shape).to(device) * epsilon / 9
    return loss_fn(outputs, targets, device)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    num = output.size(1)
    target_topk = []
    appendices = []
    for k in topk:
        if k <= num:
            target_topk.append(k)
        else:
            appendices.append([0.0])
    topk = target_topk
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res + appendices
