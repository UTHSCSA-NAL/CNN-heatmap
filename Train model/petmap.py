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
from mytools.models_MRI import ClassificationModel3D_mri0
from mytools.models_res import MyResNet
from mytools.models import DatasetFromNii, DatasetFromNii_aug, dataset_cv, train_model_PET
from torch.optim.lr_scheduler import StepLR       
import pathlib
from datetime import datetime
import sys


start_time = datetime.now()

torch.cuda.empty_cache() 


mod = sys.argv[1] 
dataset  = DatasetFromNii_aug(csv_path='data/pet_cere_'+ mod +'.csv')
out_pth = "/home/wangd2/LRP/Train/PET_experiment/" + mod + "_map/"


#dataset_aug1 = DatasetFromNii_aug(csv_path='data/Output_cere.csv', gass = 0.01)
#dataset_aug2 = DatasetFromNii_aug(csv_path='data/Output_cere.csv', gass = 0.5)
#dataset_aug3 = DatasetFromNii_aug(csv_path='data/Output_cere.csv', gass = 0.1)
#dataset_aug4 = DatasetFromNii_aug(csv_path='data/Output_cere.csv', gass = 0.05)

pathlib.Path(out_pth).mkdir(parents=True, exist_ok=True)
print(out_pth) 


b = 4 
device = int(sys.argv[2])
n_epochs = 100
patience = 10
d=0
learning_rate = 1e-4


print(f"batch size:{b}")
print(f"delta:{d}")
print(f"patience:{patience}")
print(f"learning_rate:{learning_rate}")


Train_set = torch.utils.data.Subset(dataset, list(range(0,451)))
Val_set = torch.utils.data.Subset(dataset, list(range(451,501)))


#    train0   ,_,_ = get_dataset(i, k, dataset_ori)
#    train1   ,_,_ = get_dataset(i, k, dataset_aug1)
#    train2   ,_,_ = get_dataset(i, k, dataset_aug2)
#    train3   ,_,_ = get_dataset(i, k, dataset_aug3)
#    train4   ,_,_ = get_dataset(i, k, dataset_aug4)
#    Train_set =  torch.utils.data.ConcatDataset([train0, train1, train2, train3, train4])

print(len(Train_set))
print(len(Val_set))

Train_loader = torch.utils.data.DataLoader(Train_set, batch_size=b, num_workers=0, shuffle=True)
Val_loader = torch.utils.data.DataLoader(Val_set, batch_size=b, num_workers=0, shuffle=True)

model = ClassificationModel3D_mri0().cuda(device)
loss_func = nn.CrossEntropyLoss().cuda(device)        
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

model, train_loss, valid_loss = train_model_PET(Train_loader, Val_loader, model, out_pth, device, optimizer, loss_func,  patience, n_epochs, d)

model_pth = out_pth + "state_dict.pth"
train_pth = out_pth + "train_loss.npy"
valid_pth = out_pth + "valid_loss.npy"

torch.save(model.state_dict(), model_pth)
np.save(train_pth, train_loss)
np.save(valid_pth, valid_loss)


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
 
 
 
 
 
 
