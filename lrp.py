import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
import nibabel as nib
import pandas as pd
import torch.nn.functional as F
from torch.utils import data
from mytools.innvestigator import InnvestigateModel
from mytools.models_MRI import ClassificationModel3D_mri0, ClassificationModel3D_mri0_adp
from mytools.models import DatasetFromNii,  get_ave_data,DatasetFromNii_aug
from mytools.models_res import MyResNet
import pathlib
import sys
torch.cuda.empty_cache() 

######### data #########
mod = sys.argv[1]
modality = sys.argv[3]

if modality == "pet":
	dataset = DatasetFromNii_aug(csv_path='data/pet_cere_'+ mod +'.csv')
	ex_name = "PET_experiment/"
elif modality == "mri":
	dataset = DatasetFromNii_aug(csv_path='data/t1_mean_'+ mod +'.csv')
	ex_name = "MRI_experiment/"

model_pth = "/home/wangd2/LRP/Train/" + ex_name + mod + "_map/"
out_pth =   "/home/wangd2/LRP/LRP/" + ex_name + mod + "_map/"
pathlib.Path(out_pth).mkdir(parents=True, exist_ok=True)
print(out_pth)
      
b=1
device = int(sys.argv[2])

all_target = []
all_prediction = []
all_attribution = []

model = ClassificationModel3D_mri0().cuda(device)
model.load_state_dict(torch.load(model_pth + "state_dict.pth" ))
model = model.cuda(device)
inn_model = InnvestigateModel(model, beta=0.5, method="b-rule").cuda(device)

#Train_set = torch.utils.data.Subset(dataset, list(range(0,451)))
#Val_set = torch.utils.data.Subset(dataset, list(range(451,501)))
#train_loader = DataLoader(Train_set, batch_size=b, num_workers=0, shuffle = False)

data_loader = DataLoader(dataset, batch_size=b, num_workers=0, shuffle = False)

model.eval()
for j, (data, target) in enumerate(data_loader):
	d,t = data.cuda(device), target.cuda(device)
	d,t = Variable(d), Variable(t)
	target = t.tolist()
	all_target +=target
	_, input_relevance_values = inn_model.innvestigate(in_tensor=d, rel_for_class=target)
	all_attribution += input_relevance_values.tolist()

	output = model(d)
	_, prediction = torch.max(output.data, 1)
	prediction = prediction.tolist()
	all_prediction.append(prediction)




np.save(out_pth + "/all_target.npy", all_target)
np.save(out_pth + "/all_prediction.npy", all_prediction)          
np.save(out_pth + "/all_attribution.npy", all_attribution)
print(f"all_target:{len(all_target)}")
print(f"all_prediction:{len(all_prediction)}")
print(f"all_attribution:{len(all_attribution)}")


all_target = np.load(out_pth + "/all_target.npy")
all_prediction = np.load(out_pth + "/all_prediction.npy")
all_attribution = np.load(out_pth + "/all_attribution.npy")

print("target",all_target) 
print("pred",all_prediction)

#av_TP, av_TN, av_FP, av_FN, av_Pos, av_Neg, av_ALL = get_ave_data(all_target ,all_prediction, all_attribution)

av_ALL = get_ave_data(all_target ,all_prediction, all_attribution)


img_affine = np.array([[   1.,    0.,    0.,  -96.],
                       [   0.,    1.,    0., -132.],
                       [   0.,    0.,    1.,  -78.],
                       [   0.,    0.,    0.,    1.]])

new_image = nib.Nifti1Image(np.flip(av_ALL,1), affine=img_affine)
nib.save(new_image, out_pth + "av_ALL.nii")
#new_image = nib.Nifti1Image(np.flip(av_TP,1), affine=img_affine)
#nib.save(new_image, out_pth + "av_TP.nii")
#new_image = nib.Nifti1Image(np.flip(av_TN,1), affine=img_affine)
#nib.save(new_image, out_pth + "av_TN.nii")
#new_image = nib.Nifti1Image(np.flip(av_FP,1), affine=img_affine)
#nib.save(new_image, out_pth + "av_FP.nii")
#new_image = nib.Nifti1Image(np.flip(av_FN,1), affine=img_affine)
#nib.save(new_image, out_pth + "av_FN.nii")
#new_image = nib.Nifti1Image(np.flip(av_Pos,1), affine=img_affine)
#nib.save(new_image, out_pth + "av_Pos.nii")
#new_image = nib.Nifti1Image(np.flip(av_Neg,1), affine=img_affine)
#nib.save(new_image, out_pth + "av_Neg.nii")



############################################ MyResNet ############################################    
    

#i = 1 
#
#fold_size = len(dataset) // 10
#idx = list(range(len(dataset))) # idx = [0, 1, 2,....500]
#
#Test_idx = idx[i* fold_size :i * fold_size + fold_size]  
#Test_set = Subset(dataset, Test_idx)
#Test_loader = DataLoader(Test_set, batch_size=b, num_workers=0, shuffle = True)
#
#device = 0
#model = MyResNet().cuda(device)
#model.load_state_dict(torch.load("/home/wangd2/LRP/Train/PET_experiment/model_res/state_dict_f" + str(i) +".pth"))
#model = model.cuda(device)
#inn_model = InnvestigateModel(model, beta=0.5, method="b-rule").cuda(device) 
# 
#all_target, all_prediction, all_relevence=get_relevence(Test_loader)
#np.save("out_model_res/all_target_" + str(i) +".npy", all_target)
#np.save("out_model_res/all_prediction_" + str(i) +".npy", all_prediction)          
#np.save("out_model_res/all_relevence_" + str(i) +".npy", all_relevence) 

















   
    
