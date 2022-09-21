import math
import numpy as np 

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.optim.lr_scheduler import _LRScheduler


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def check_graph(data, piece=2, THRESHOLD=None, col=None, LOSS_img=False):
    
    if LOSS_img == True:
        real = np.array(data['loss'])
        
        region = ranges(list(np.where(real >= THRESHOLD)[0]))
        
        l1 = real.shape[0]
            
        chunk = l1 // piece
        fig, axs = plt.subplots(piece, figsize=(30, 4 * piece))
        
        for i in range(piece):
            L = i * chunk
            R1 = min(L + chunk, l1)
            
            xticks = range(L, R1)
            
            if piece == 1:
                plt.plot(xticks, real[L:R1], label="Loss")
                for re in region:
                    plt.axvspan(re[0], re[1], facecolor='g', alpha=0.3)

            else:  
                axs[i].plot(xticks, real[L:R1], label="Loss")
                for re in region:
                    plt.axvspan(re[0], re[1], facecolor='g', alpha=0.3)
                
            if len(real[L:R1]) > 0:
                peak = np.max(real[L:R1])
                                
            if piece == 1:
                plt.axhline(y=THRESHOLD, color='r')
            else:
                axs[i].axhline(y=THRESHOLD, color='r')
        plt.legend()
        plt.show()
        
    else:
        real = np.array(data[col])
        loss = np.array(data['loss'])
    
        region = ranges(list(np.where(loss >= THRESHOLD)[0]))
        
        l1 = real.shape[0]
        
        chunk = l1 // piece
        fig, axs = plt.subplots(piece, figsize=(30, 4 * piece))
        
        for i in range(piece):
            L = i * chunk
            R1 = min(L + chunk, l1)
            R2 = min(L + chunk, l1)
            
            xticks = range(L, R1)
            
            if piece == 1:
                plt.plot(xticks, real[L:R1], label="Real")
                for re in region:
                    plt.axvspan(re[0], re[1], facecolor='g', alpha=0.3)

            else:  
                axs[i].plot(xticks, real[L:R1], label="Real")
                for re in region:
                    plt.axvspan(re[0], re[1], facecolor='g', alpha=0.3)
                
            if len(real[L:R1]) > 0:
                peak = np.max(real[L:R1])
            plt.title('{}'.format(col))
                    
        plt.legend()
        plt.show()
    

def plot_history(history, idx, config):
    # plt.figure(figsize=(12,10))
    losses1 = history['valid_loss1']
    losses2 = history['valid_loss2']
    plt.plot(losses1, '-x', label="AE1 loss")
    plt.plot(losses2, '-x', label="AE2 loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.axvline(x=idx, color='r', linestyle='--')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.savefig(config['model_save_name'] + ".png")
    plt.show()
    

class CustomDataset(Dataset):
    def __init__(self, data):
        super(CustomDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        return record


class BucketBatchingSampler(Sampler):
    def __init__(self, data_source: Dataset, config: None):
        super(BucketBatchingSampler, self).__init__(data_source)
        self.data_source = data_source      
        self.config = config      

        ids = list(range(len(self.data_source)))
        self.bins = [ids[i:i + self.config['batch_size']] for i in range(0, len(ids), self.config['batch_size'])]
                
    def __iter__(self):
        for ids in self.bins:
            yield ids

    def __len__(self):
        return len(self.bins)


class Custom_collate_fn(object):
    def __init__(self, config):
        self.config = config
    
    def __call__(self, batch):
        return batch


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def criterion(real, w1, w2, w3, n):
    loss1 = 1/n*torch.mean((real-w1)**2) + (1-1/n)*torch.mean((real-w3)**2)
    loss2 = 1/n*torch.mean((real-w2)**2) - (1-1/n)*torch.mean((real-w3)**2)

    return loss1, loss2


def calc_missing(df, col):
    return df[col].isnull().sum() / len(df[col])


def press_to_alt(x):
    return ((10 ** (math.log10((x / 101325)) / 5.2558797) - 1) / (-6.875586 * math.pow(10,-6))) * 0.3048 


def convert_week(x):
    day = int(x.split("-")[-1].split(" ")[0])

    value = 1 if day <=7 else 2 if day <=14 else 3 if day <=21 else 4 if day <=28 else 5
    return value


def dummy_and_add_feature(x):
    hour = int(x.split(" ")[-1].split(":")[0])
    day = int(x.split("-")[-1].split(" ")[0])
    month = int(x.split("-")[1])
    
    sin_hour = np.sin((2*np.pi*hour*60*60)/(24*60*60))
    cos_hour = np.cos((2*np.pi*hour*60*60)/(24*60*60))
    sin_day = np.sin((2*np.pi*day*24*60*60)/(31*24*60*60))
    cos_day = np.cos((2*np.pi*day*24*60*60)/(31*24*60*60))
    sin_month = np.sin((2*np.pi*month*31*24*60*60)/(12*31*24*60*60))
    cos_month = np.cos((2*np.pi*month*31*24*60*60)/(12*31*24*60*60))
    
    return sin_hour, cos_hour, sin_day, cos_day, sin_month, cos_month
