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
from mytools.models import dataset_cv_82, get_model, DatasetFromNii, DatasetFromNii_aug, dataset_cv, train_model_MRI,train_model_PET
import pathlib
from datetime import datetime
import sys


torch.cuda.empty_cache() 
fold = sys.argv[1]
mod = sys.argv[2]
which_model = sys.argv[3]
modality = sys.argv[4]

if modality == "PET":
	prex="pet_cere"
elif modality == "MRI":
	prex = "t1_mean"

if mod == "aug":
	dataset_ori  = DatasetFromNii_aug(csv_path='data/' + prex + '_ori.csv')
	dataset_aug1 = DatasetFromNii_aug(csv_path='data/' + prex + '_ori.csv', gass = 0.01)
	dataset_aug2 = DatasetFromNii_aug(csv_path='data/' + prex + '_ori.csv', gass = 0.5)
	dataset_aug3 = DatasetFromNii_aug(csv_path='data/' + prex + '_ori.csv', gass = 0.1)
	dataset_aug4 = DatasetFromNii_aug(csv_path='data/' + prex + '_ori.csv', gass = 0.05)
	out_pth = "/home/wangd2/LRP/Train/" + modality +"_experiment/aug/" + which_model + "/" 
elif mod == "augmp":
	dataset_ori  = DatasetFromNii_aug(csv_path='data/' + prex + '_mprage.csv')
	dataset_aug1 = DatasetFromNii_aug(csv_path='data/' + prex + '_mprage.csv', gass = 0.01)
	dataset_aug2 = DatasetFromNii_aug(csv_path='data/' + prex + '_mprage.csv', gass = 0.5)
	dataset_aug3 = DatasetFromNii_aug(csv_path='data/' + prex + '_mprage.csv', gass = 0.1)
	dataset_aug4 = DatasetFromNii_aug(csv_path='data/' + prex + '_mprage.csv', gass = 0.05)
	out_pth = "/home/wangd2/LRP/Train/" + modality +"_experiment/augmp/" + which_model + "/" 
elif mod == "ori":
	dataset_ori  = DatasetFromNii_aug(csv_path='data/'+ prex +'_ori.csv')
	out_pth = "/home/wangd2/LRP/Train/"+ modality +"_experiment/ori/" + which_model + "/"
elif mod == "mprage":
	dataset_ori  = DatasetFromNii_aug(csv_path='data/'+ prex +'_mprage.csv')
	out_pth = "/home/wangd2/LRP/Train/"+ modality +"_experiment/mprage/" + which_model + "/"
elif mod == "jacobian":
	dataset_ori  = DatasetFromNii_aug(csv_path='data/'+ prex +'_jacobian.csv', minmax = True)
	out_pth = "/home/wangd2/LRP/Train/"+ modality +"_experiment/jacobian/" + which_model + "/"
elif mod.startswith("S"):
	dataset_ori  = DatasetFromNii_aug(csv_path='data/'+ prex +'_' + mod + '.csv')
	out_pth = "/home/wangd2/LRP/Train/"+ modality +"_experiment/" + mod + "/" + which_model + "/"



b = 4
k = 5        
device = 0
n_epochs = 100
patience = 10
d=0
learning_rate = 1e-4

pathlib.Path(out_pth).mkdir(parents=True, exist_ok=True)
print(out_pth)



for i in [int(fold)]:
    torch.cuda.empty_cache() 
   
    if mod == "aug":
        train0   ,_,_ = dataset_cv(i, k, dataset_ori)
        train1   ,_,_ = dataset_cv(i, k, dataset_aug1)
        train2   ,_,_ = dataset_cv(i, k, dataset_aug2)
        train3   ,_,_ = dataset_cv(i, k, dataset_aug3)
        train4   ,_,_ = dataset_cv(i, k, dataset_aug4)
        Train_set =  torch.utils.data.ConcatDataset([train0, train1, train2, train3, train4])
    else :
        Train_set,_,_ = dataset_cv(i, k, dataset_ori)
    
    _,Val_set,_ = dataset_cv(i, k,dataset_ori)
    
    Train_loader = torch.utils.data.DataLoader(Train_set, batch_size=b, num_workers=0, shuffle=True)
    Val_loader = torch.utils.data.DataLoader(Val_set, batch_size=b, num_workers=0, shuffle=True)
    
    model = get_model(which_model, device)
    loss_func = nn.CrossEntropyLoss().cuda(device)        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    if modality == "MRI":
        model, train_loss, valid_loss = train_model_MRI(Train_loader, Val_loader, model, out_pth, device, optimizer, loss_func, patience, n_epochs, d, i)
    elif modality == "PET":
        model, train_loss, valid_loss = train_model_PET(Train_loader, Val_loader, model, out_pth, device, optimizer, loss_func, patience, n_epochs, d, i)
    model_pth = out_pth + "state_dict_f" + str(i) +".pth"
    train_pth = out_pth + "train_loss_f" + str(i) +".npy"
    valid_pth = out_pth + "valid_loss_f" + str(i) +".npy"
    torch.save(model.state_dict(), model_pth)
    np.save(train_pth, train_loss)
    np.save(valid_pth, valid_loss)
 
