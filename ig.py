import torch
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
from mytools.innvestigator import InnvestigateModel
from mytools.models_MRI import ClassificationModel3D_mri0, ClassificationModel3D_mri0_adp
from mytools.models import DatasetFromNii, get_ave_data, DatasetFromNii_aug
from mytools.models_res import MyResNet, MyResNet0
import pathlib
import sys
torch.cuda.empty_cache() 

def calculate_outputs_and_gradients(inputs, model, cuda=True):
    gradients = []

    for input in inputs:  # input = torch.Size([1, 193, 229, 193])
        input = torch.tensor(input, dtype=torch.float32, device=device, requires_grad=True)
        output = model(input)      
        
        index = np.ones((output.size()[0], 1)) * target.item()       #  np.ones((output.size()[0], 1)) =  array([[1.]])
        index = torch.tensor(index, dtype=torch.int64)
        index = index.cuda()
        output = output.gather(1, index)         # just get the max results ex. output = tensor([[0.9068]])

        # clear grad
        model.zero_grad()
        output.backward()
        gradient = input.grad.detach().cpu().numpy()[0]
        gradients.append(gradient)
        
    gradients = np.array(gradients)
    
    return gradients


def integrated_gradients(inputs, model, baseline_name, steps, cuda=True):
    
    if baseline_name == "z":
        baseline = 0 * inputs
    if baseline_name == "u":
        baseline = torch.rand(1, 1, 193, 229, 193)                 # [0, 1]   
    if baseline_name == "g":
        baseline = inputs + torch.randn(1, 1, 193, 229, 193)       # gaussian
    if baseline_name == "a":
        baseline = torch.rand(1, 1, 193, 229, 193) + inputs + torch.randn(1, 1, 193, 229, 193)       # average u and g
        baseline = baseline/2
     
    scaled_inputs = [baseline + (float(k) / steps) *( inputs - baseline) for k in range(0, steps + 1)] # ([51，1, 1, 193, 229, 193])
   
    grads = calculate_outputs_and_gradients(scaled_inputs, model,cuda) 
    
    #avg_grads = np.absolute(np.average(grads[:-1], axis=0))    
    avg_grads = np.average(grads[:-1], axis=0)                            # average image gradient
    
    inputs_torch = torch.tensor(inputs, dtype=torch.float32, device=device, requires_grad=True)
    baseline_torch = torch.tensor(baseline, dtype=torch.float32, device=device, requires_grad=True)
    delta_X = (inputs_torch - baseline_torch).detach().squeeze(0).cpu().numpy()
    integrated_grad = delta_X * avg_grads
    
    return integrated_grad




############ data #################

mod = sys.argv[1]
modality = sys.argv[3]
      
if modality == "pet":
	dataset = DatasetFromNii_aug(csv_path='data/pet_cere_'+ mod +'.csv')
	ex_name = "PET_experiment/"
elif modality == "mri":
	dataset = DatasetFromNii_aug(csv_path='data/t1_mean_'+ mod +'.csv')
	ex_name = "MRI_experiment/"
      
model_pth = "/home/wangd2/LRP/Train/" + ex_name + mod + "_map/"
out_pth =   "/home/wangd2/LRP/IG/" + ex_name + mod + "_map/"
pathlib.Path(out_pth).mkdir(parents=True, exist_ok=True)
print(out_pth)



b = 1
device = int(sys.argv[2])
name = "z"
all_attribution = []
all_prediction = []
all_target = []

model = ClassificationModel3D_mri0().cuda(device)
model.load_state_dict(torch.load(model_pth + "state_dict.pth"))
model = model.cuda(device)
model.eval()

data_loader = DataLoader(dataset, batch_size=b, num_workers=0, shuffle = False)

for j, (data, target) in enumerate(data_loader):     # data = torch.Size([1, 1, 193, 229, 193])
    all_target += target.tolist()  # tensor to list
    d = data.cuda(device)
    d = Variable(d)
    output=model(d)
    _, prediction = torch.max(output.data, 1)
    prediction = prediction.cpu().detach().numpy()
    all_prediction += prediction.tolist()

    attributions = integrated_gradients(data, model, baseline_name = name, steps=50, cuda=True)
    all_attribution.append(attributions) # numpy can append

print(f"all_attributions:{len(all_attribution)}")
print(f"all_prediction:{len(all_prediction)}")
print(f"all_target: {len(all_target)}")

np.save(out_pth + "/all_target.npy", all_target)
np.save(out_pth + "/all_prediction.npy", all_prediction)
np.save(out_pth + "/all_attribution.npy", all_attribution)


################## get ave data


#for i in range(k):
#    if i == 0:
#        i_target = np.load(out_pth + "all_target" + str(i)+ ".npy")
#        i_prediction = np.load(out_pth + "all_prediction" + str(i)+ ".npy")
#        i_attribution = np.load(out_pth + "all_attribution" + str(i)+ ".npy")
#    else:
#        fold_target = np.load(out_pth + "all_target" + str(i)+ ".npy")
#        fold_prediction = np.load(out_pth + "all_prediction" + str(i)+ ".npy")
#        fold_attribution = np.load(out_pth + "all_attribution" + str(i)+ ".npy")
#
#        i_target = np.concatenate((i_target, fold_target),axis=0)
#        i_prediction = np.concatenate((i_prediction, fold_prediction),axis=0)
#        i_attribution = np.concatenate((i_attribution, fold_attribution),axis=0)

i_target = np.load(out_pth + "all_target.npy")
i_prediction = np.load(out_pth + "all_prediction.npy")
i_attribution = np.load(out_pth + "all_attribution.npy")

print(i_target.shape)
print(i_prediction.shape)
print(i_attribution.shape)


av_ALL = get_ave_data(i_target ,i_prediction, i_attribution)

img_affine = np.array([[   1.,    0.,    0.,  -96.],
                       [   0.,    1.,    0., -132.],
                       [   0.,    0.,    1.,  -78.],
                       [   0.,    0.,    0.,    1.]])

new_image = nib.Nifti1Image(np.flip(av_ALL,1), affine=img_affine)
nib.save(new_image, out_pth + "av_ALL.nii")


    
    
    
    
