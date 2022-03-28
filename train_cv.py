import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import nibabel as nib
import pandas as pd
from torch.utils import data
from mytools.utils import *
import pathlib
from datetime import datetime
import sys
import argparse
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--path', type=str, help='path to csv file of dataset')
parser.add_argument('--outpath', type=str, help='path to save model')
parser.add_argument('--device', type=int, help='gpu device')
parser.add_argument('--model', type=str, help='name of model')
parser.add_argument('--batch', default=5, type=int, help='batch size (default: 5)')
parser.add_argument('--fold', type=int, default=5, help='fold for CV (default: 5)')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate (default: 1e-4)')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=200,help='number of total epochs to run (default: 200)')
parser.add_argument('--patience', default=10, type=int, help='patience used for early stop rate (default: 10)')
parser.add_argument('--delta', default=0, type=float, help='delta used for early stop rate (default: 0)')
args = parser.parse_args()

dataset_ori  = DatasetFromNii_aug(csv_path=args.path, minmax=True)
out_pth = args.outpath
pathlib.Path(out_pth).mkdir(parents=True, exist_ok=True)

##### train model for each fold
for i in range(0, args.fold):
    torch.cuda.empty_cache() 
    Train_set,_,_ = dataset_cv(i, args.fold, dataset_ori)
    _,Val_set,_ = dataset_cv(i, args.fold,dataset_ori)
    Train_loader = torch.utils.data.DataLoader(Train_set, batch_size = args.batch, num_workers=0, shuffle=True)
    Val_loader = torch.utils.data.DataLoader(Val_set, batch_size = args.batch, num_workers=0, shuffle=True)
    
    model = get_model(args.model, args.device)
    loss_func = nn.CrossEntropyLoss().cuda(args.device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
    model, train_loss, valid_loss = train_earlystop(Train_loader, Val_loader, model, out_pth, args.batch, optimizer, loss_func, args.patience, args.epochs, args.delta, i)

    model_pth = out_pth + "state_dict_f" + str(i) +".pth"
    train_pth = out_pth + "train_loss_f" + str(i) +".npy"
    valid_pth = out_pth + "valid_loss_f" + str(i) +".npy"
    torch.save(model.state_dict(), model_pth)
    np.save(train_pth, train_loss)
    np.save(valid_pth, valid_loss)
 
