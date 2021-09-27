import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd
import torch.nn.functional as F
from torch.utils import data
from mytools.models_MRI import ClassificationModel3D_mri0
from mytools.models import DatasetFromNii, get_ave_data, DatasetFromNii_aug
from captum.attr import GuidedGradCam
from mytools.models_res import MyResNet
import pathlib
import sys
torch.cuda.empty_cache() 

l_number = 16  #16 12 8


######### or #########                                                           
mod = sys.argv[1]
modality = sys.argv[3]
  
if modality == "pet":
    dataset = DatasetFromNii_aug(csv_path='data/pet_cere_'+ mod +'.csv')
    ex_name = "PET_experiment/"
elif modality == "mri":
    dataset = DatasetFromNii_aug(csv_path='data/t1_mean_'+ mod +'.csv')
    ex_name = "MRI_experiment/"
  
model_pth = "/home/wangd2/LRP/Train/" + ex_name + mod + "_map/"
out_pth =   "/home/wangd2/LRP/GGC/" + ex_name + mod + "_map/"
pathlib.Path(out_pth).mkdir(parents=True, exist_ok=True)
print(out_pth)


device = int(sys.argv[2])
b = 1
all_prediction = []
all_target = []
all_attribution = []


model = ClassificationModel3D_mri0().cuda(device)
model.load_state_dict(torch.load(model_pth + "state_dict.pth"))
model = model.cuda(device)

guided_gc = GuidedGradCam(model, layer=model.features[l_number])
data_loader = DataLoader(dataset, batch_size=b, num_workers=0, shuffle = False)

for j, (data, target) in enumerate(data_loader):     # data = torch.Size([1, 1, 193, 229, 193])
      model.eval()

      torch.cuda.empty_cache()
      all_target += target.tolist()  # tensor to list
      d = data.cuda(device)
      d = Variable(d)

      output = model(d)
      _, prediction = torch.max(output.data, 1)
      prediction = prediction.cpu().detach().numpy()
      all_prediction += prediction.tolist()

      attribution = guided_gc.attribute(d, target = target, interpolate_mode = "trilinear")
      all_attribution.append(attribution.cpu().detach().numpy())


print(f"all_prediction:{all_prediction}")
print(f"all_target: {all_target}")

np.save(out_pth + "/all_prediction.npy", all_prediction)
np.save(out_pth + "/all_target.npy", all_target)
np.save(out_pth + "/all_attribution.npy", all_attribution)


all_target = np.load(out_pth + "/all_target.npy")
all_prediction = np.load(out_pth + "/all_prediction.npy")
all_attribution = np.load(out_pth + "/all_attribution.npy")
all_attribution = np.squeeze(all_attribution, 1)
print(all_attribution.shape)


av_ALL = get_ave_data(all_target ,all_prediction, all_attribution)

img_affine = np.array([[   1.,    0.,    0.,  -96.],
                       [   0.,    1.,    0., -132.],
                       [   0.,    0.,    1.,  -78.],
                       [   0.,    0.,    0.,    1.]])

new_image = nib.Nifti1Image(np.flip(av_ALL,1), affine=img_affine)
nib.save(new_image, out_pth + "av_ALL.nii")
    
    
