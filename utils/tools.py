import time
import random
import numpy as np

import matplotlib.pyplot as plt
import os.path as osp

import torch
import torch.nn.functional as F
import torch.distributed as dist


def set_deterministic(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_custom_figure(epoch, data_dict, figure_title, figure_dir, y_lim=None):
    plt.figure()
    epoch = list(range(1, epoch + 1))
    plt.xlabel('epoch')
    if y_lim:
        plt.ylim(y_lim)
    for k, v in data_dict.items():
        plt.plot(epoch, v, label=k)
    plt.legend()
    plt.title(figure_title)
    plt.savefig(osp.join(figure_dir, f'{figure_title}.png'))
    plt.close()


class Timer(object):
    def __init__(self):
        self.on = False
        self.begin_t = None
        self.end_t = None

    def begin(self):
        self.on = True
        self.begin_t = time.time()

    def end(self):
        if not self.on:
            raise Exception('timer is not on!')
        self.end_t = time.time()

    def time_str(self):
        t = self.end_t - self.begin_t
        h, m, s = t // 3600, (t % 3600) // 60, (t % 3600) % 60
        return f'Elapsed Time: {h:.0f}h {m:.0f}m {s:.2f}s'


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count]).cuda()
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return ' Batch [' + fmt + '/' + fmt.format(num_batches) + ']'
