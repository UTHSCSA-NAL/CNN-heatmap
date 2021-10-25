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
from mytools.earlystop_PET import EarlyStopping
from mytools.models_MRI import unicnn_f16l5,unicnn_f16l4,unicnn_f16l3, ClassificationModel3D_mri0, ClassificationModel3D_mri0_4,ClassificationModel3D_mri0_u,ClassificationModel3D_mri0_3
from mytools.models_res import MyResNet
from mytools.models import get_model,DatasetFromNii, DatasetFromNii_aug, dataset_cv, train_model_MRI
from torch.optim.lr_scheduler import StepLR       
import pathlib
from datetime import datetime
import sys

start_time = datetime.now()
torch.cuda.empty_cache() 

mod = sys.argv[1]
which_model = sys.argv[2]

b = 4
if mod == "aug":
	dataset  = DatasetFromNii_aug(csv_path='data/t1_mean_ori.csv')
	dataset_aug1 = DatasetFromNii_aug(csv_path='data/t1_mean_ori.csv', gass = 0.01)
	dataset_aug2 = DatasetFromNii_aug(csv_path='data/t1_mean_ori.csv', gass = 0.5)
	dataset_aug3 = DatasetFromNii_aug(csv_path='data/t1_mean_ori.csv', gass = 0.1)
	dataset_aug4 = DatasetFromNii_aug(csv_path='data/t1_mean_ori.csv', gass = 0.05)
	train0 = torch.utils.data.Subset(dataset, list(range(0,451)))
	train1 = torch.utils.data.Subset(dataset_aug1, list(range(0,451)))
	train2 = torch.utils.data.Subset(dataset_aug2, list(range(0,451)))
	train3 = torch.utils.data.Subset(dataset_aug3, list(range(0,451)))
	train4 = torch.utils.data.Subset(dataset_aug4, list(range(0,451)))
	Train_set =  torch.utils.data.ConcatDataset([train0, train1, train2, train3, train4])
	Train_loader = torch.utils.data.DataLoader(Train_set, batch_size=b, num_workers=0, shuffle=True)
	Val_set = torch.utils.data.Subset(dataset, list(range(451,501)))
	Val_loader = torch.utils.data.DataLoader(Val_set, batch_size=b, num_workers=0, shuffle=True) 
else:
	dataset  = DatasetFromNii_aug(csv_path='data/t1_mean_'+ mod +'.csv')
	Train_set = torch.utils.data.Subset(dataset, list(range(0,451)))
	Val_set = torch.utils.data.Subset(dataset, list(range(451,501)))
	Train_loader = torch.utils.data.DataLoader(Train_set, batch_size=b, num_workers=0, shuffle=True)
	Val_loader = torch.utils.data.DataLoader(Val_set, batch_size=b, num_workers=0, shuffle=True)
  

device = 0
n_epochs = 100
patience = 10
d=0
learning_rate = 1e-4
                             
out_pth = "/home/wangd2/LRP/Train/MRI_experiment/" + mod + "_map/"+ which_model + "/"
pathlib.Path(out_pth).mkdir(parents=True, exist_ok=True)
print(out_pth)

model = get_model(which_model, device)
loss_func = nn.CrossEntropyLoss().cuda(device)        
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

model, train_loss, valid_loss = train_model_MRI(Train_loader, Val_loader, model, out_pth, device, optimizer, loss_func,  patience, n_epochs, d, i="map")

model_pth = out_pth + "state_dict.pth"
train_pth = out_pth + "train_loss.npy"
valid_pth = out_pth + "valid_loss.npy"

torch.save(model.state_dict(), model_pth)
np.save(train_pth, train_loss)
np.save(valid_pth, valid_loss)


 

