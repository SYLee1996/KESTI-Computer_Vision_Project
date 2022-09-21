import os
import math
import copy 
import pickle
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader

from SURFACE_UTILS import (CustomDataset, BucketBatchingSampler, Custom_collate_fn, CosineAnnealingWarmUpRestarts,
                            to_device, criterion, EarlyStopping, plot_history)
from SURFACE_MODEL import Network

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Training', add_help=False)

    # Model parameters
    parser.add_argument('--batch_size', default=1176, type=int)
    parser.add_argument('--window_size', default=24, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)


    # Optimizer parameters
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_t', default=10, type=int)
    parser.add_argument('--lr_scheduler', default='CosineAnnealingLR', type=str)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)


    # Training parameters
    parser.add_argument('--train_data', default='data_path', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--text', default='default', type=str)
    parser.add_argument('--device', default='0,1,2,3', type=str)

    return parser


def main(args):

    seed = 10
    suffix = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M")

    config = {
        # Model parameters
        'batch_size': args.batch_size,
        'window_size': args.window_size,
        'hidden_size': args.hidden_size,
        
        # Optimizer parameters
        'optimizer': args.optimizer,
        'lr': args.lr,
        'lr_t': args.lr_t,
        'lr_scheduler': args.lr_scheduler,
        'gamma': args.gamma,
        'patience': args.patience,
        'weight_decay': args.weight_decay,
        
        # Training parameters
        'train_data': args.train_data,
        'epochs': args.epochs,
        'num_workers': args.num_workers,
        'text': args.text,
        'device': args.device,
        }
    
    model_save_name='./RESULTS/'+config['text']+"_"+suffix+"("+ str(config['train_data'].split("_")[-1].split(".")[0])+"_"+\
                                                                str(config['batch_size'])+"_"+\
                                                                str(config['window_size'])+"_"+\
                                                                str(config['hidden_size'])+"___"+\
                                                                str(config['optimizer'])+"_"+\
                                                                str(config['lr'])+"_"+\
                                                                str(config['lr_t'])+"_"+\
                                                                str(config['lr_scheduler'])+"_"+\
                                                                str(config['gamma'])+"_"+\
                                                                str(config['patience'])+"_"+\
                                                                str(config['weight_decay'])+"___"+\
                                                                str(config['epochs'])+")"
                                                            
    config['model_save_name'] = model_save_name
    print('model_save_name: '+config['model_save_name'].split("/")[-1])
    # -------------------------------------------------------------------------------------------

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    config['device'] = device

    # -------------------------------------------------------------------------------------------
    
    # Data load
    Train_df = pd.read_csv(config['train_data'])

    # 개수가 맞지 않는 stnID를 삭제 (train에선 4704)
    ddd = pd.DataFrame(Train_df.stnID.value_counts() != 4704)
    del_list = ddd.index[ddd['stnID'] == True].tolist()
    Train_df = Train_df[~Train_df['stnID'].isin(del_list)].sort_values(["stnID", "Date/Time"]).reset_index(drop=True)

    # Train, Valid split by each station
    valid_stn = random.sample(list(Train_df.stnID.unique()), int(len(Train_df.stnID.unique()) * 0.1))

    Valid_df = Train_df[Train_df['stnID'].isin(valid_stn)].sort_values(["stnID", "Date/Time"]).reset_index(drop=True)
    Train_df = Train_df[~Train_df['stnID'].isin(valid_stn)].sort_values(["stnID", "Date/Time"]).reset_index(drop=True)

    Train_df_stnID_DateTime = Train_df[['Date/Time', 'stnID']]
    Valid_df_stnID_DateTime = Valid_df[['Date/Time', 'stnID']]

    Train_df.drop(columns=['Date/Time', 'stnID'], inplace=True)
    Valid_df.drop(columns=['Date/Time', 'stnID'], inplace=True)

    windows_train=Train_df.values[np.arange(config['window_size'])[None, :] + np.arange(Train_df.shape[0]-config['window_size']+1)[:, None]]
    windows_valid=Valid_df.values[np.arange(config['window_size'])[None, :] + np.arange(Valid_df.shape[0]-config['window_size']+1)[:, None]]

    config['w_size'] = windows_train.shape[1] * windows_train.shape[2]
    config['z_size'] = windows_train.shape[1] * config['hidden_size']

    torch_train = torch.from_numpy(windows_train).float().view(([windows_train.shape[0], config['w_size']]))
    torch_valid = torch.from_numpy(windows_valid).float().view(([windows_valid.shape[0], config['w_size']]))

    # -------------------------------------------------------------------------------------------
    # Train
    train_set = CustomDataset(data=torch_train)
    sampler = BucketBatchingSampler(data_source=train_set, config=config)
    collate_fn = Custom_collate_fn(config=config)
    Train_loader=DataLoader(dataset=train_set,
                            sampler=sampler,
                            collate_fn=collate_fn,
                            pin_memory=False, num_workers=config['num_workers'], 
                            )

    # Valid
    valid_set = CustomDataset(data=torch_valid)
    sampler = BucketBatchingSampler(data_source=valid_set, config=config)
    collate_fn = Custom_collate_fn(config=config)
    Valid_loader=DataLoader(dataset=train_set,
                            sampler=sampler,
                            collate_fn=collate_fn,
                            pin_memory=False, num_workers=config['num_workers'], 
                            )

    model = Network(config).to(config['device'])
    model = nn.DataParallel(model).to(config['device'])

    if config['lr_scheduler'] == 'CosineAnnealingLR':
        optimizer1=torch.optim.AdamW(list(model.module.encoder.parameters())+list(model.module.decoder1.parameters()), 
                                    lr=config['lr'],
                                    weight_decay=config['weight_decay'],)
        optimizer2=torch.optim.AdamW(list(model.module.encoder.parameters())+list(model.module.decoder2.parameters()), 
                                    lr=config['lr'],
                                    weight_decay=config['weight_decay'],)
                                
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=config['lr_t'], eta_min=0)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=config['lr_t'], eta_min=0)
    
    elif config['lr_scheduler'] == 'CosineAnnealingWarmUpRestarts':
        optimizer1 = torch.optim.AdamW(list(model.module.encoder.parameters())+list(model.module.decoder1.parameters()), lr=0)
        optimizer2 = torch.optim.AdamW(list(model.module.encoder.parameters())+list(model.module.decoder1.parameters()), lr=0)
        
        scheduler1 = CosineAnnealingWarmUpRestarts(optimizer1, T_0=config['lr_t'], eta_max=config['lr'], gamma=config['gamma'], T_mult=1, T_up=0)
        scheduler2 = CosineAnnealingWarmUpRestarts(optimizer2, T_0=config['lr_t'], eta_max=config['lr'], gamma=config['gamma'], T_mult=1, T_up=0)

    
    early_stopping_loss = EarlyStopping(patience=config['patience'], mode='min')
    
    # -------------------------------------------------------------------------------------------
    epochs = config['epochs']
    history = { 'valid_loss1':[],
                'valid_loss2':[]}
    
    best_loss = 10
    mean_loss_treshold_list = []
    sum_of_losses = [10]
    
    for epoch in range(epochs):
        valid_loss1, valid_loss2 = 0, 0
        loss_treshold_list = []
                
        model.train()
        tqdm_dataset = tqdm(enumerate(Train_loader), total=len(Train_loader))

        for batch_id, [batch] in tqdm_dataset:
            batch = to_device(batch, config['device'])
            
            #Train AE1
            w1, w2, w3 = model(batch)
            loss1, loss2 = criterion(batch, w1, w2, w3, epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            #Train AE2
            w1, w2, w3 = model(batch)
            loss1, loss2 = criterion(batch, w1, w2, w3, epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
            tqdm_dataset.set_postfix({
                'Epoch' : epoch+1,
                'loss1' : '{:06f}'.format(loss1),
                'loss2' : '{:06f}'.format(loss2),
                })
            
        scheduler1.step()
        scheduler2.step()
        
        model.eval()
        tqdm_valid_dataset = tqdm(enumerate(Valid_loader), total=len(Valid_loader))
        
        for val_batch_id, [val_batch] in tqdm_valid_dataset:
            val_batch = to_device(val_batch, config['device'])
            
            #valid
            val_w1, val_w2, val_w3 = model(val_batch)
            val_loss1, val_loss2 = criterion(val_batch, val_w1, val_w2, val_w3, epoch+1)
            loss_treshold_list.append(0.5*torch.mean((val_batch-val_w1)**2,axis=1)+0.5*torch.mean((val_batch-val_w2)**2,axis=1))
            
            valid_loss1 += val_loss1.item()
            valid_loss2 += val_loss2.item()
            
            tqdm_valid_dataset.set_postfix({
                'Epoch' : epoch+1,
                'loss1' : '{:06f}'.format(valid_loss1),
                'loss2' : '{:06f}'.format(valid_loss2),
                })
                
        mean_loss_treshold_list.append(np.quantile(
            np.concatenate([torch.stack(loss_treshold_list[:-1]).flatten().detach().cpu().numpy(), 
                                        loss_treshold_list[-1].flatten().detach().cpu().numpy()]), 0.75))
        
        # -------------------------------------------------------------------------------------------
        valid_loss1 = valid_loss1/len(Valid_loader)
        valid_loss2 = valid_loss2/len(Valid_loader)    
    
        history['valid_loss1'].append(valid_loss1)
        history['valid_loss2'].append(valid_loss2)
        
        print_best = 0    
        if (history['valid_loss1'][-1] <= best_loss):
            best_loss = history['valid_loss1'][-1]
            
            best_idx = epoch+1
            model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict()
            best_model_wts = copy.deepcopy(model_state_dict)
            
            # load and save best model weights
            model.module.load_state_dict(best_model_wts)
            torch.save(best_model_wts, config['model_save_name'] + ".pt")
            print_best = '==> best model saved %d epoch / sum of loss1 & loss2 : %.5f'%(best_idx, history['valid_loss1'][-1]+history['valid_loss2'][-1])
        
        sum_of_losses.append(history['valid_loss1'][-1]+history['valid_loss2'][-1])
        del loss_treshold_list
        print(f'Epoch [{epoch+1}], val_loss1: {valid_loss1:.4f}, val_loss2: {valid_loss2:.4f}')
        print('\n') if type(print_best)==int else print(print_best,'\n')
        
        if (epoch>=50) & (early_stopping_loss.step(torch.tensor(history['valid_loss1'][-1]))):
        # if (epoch>=300) & (early_stopping_loss.step(torch.tensor(history['valid_loss1'][-1]))):
            break
        if (epoch>=100) & (abs(sum_of_losses[-1]-sum_of_losses[-2]) <= 1e-6):
        # if (epoch>=350) & (abs(sum_of_losses[-1]-sum_of_losses[-2]) <= 1e-6):
            break
        
    plot_history(history, best_idx-1, config)
    print(config['model_save_name'].split("/")[-1] + ' is saved!')
    print('Valid loss 75% quantile is {}!'.format(np.mean(mean_loss_treshold_list)))

    file = open('{}.txt'.format(config['model_save_name']), 'w')
    file.write(str(np.mean(mean_loss_treshold_list)))
    file.close()
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)


