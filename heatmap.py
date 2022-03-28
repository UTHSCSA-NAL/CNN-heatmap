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
from mytools.utils import *
from mytools.ig import *
from mytools.innvestigator import InnvestigateModel
from captum.attr import GuidedGradCam
import pathlib
import sys

torch.cuda.empty_cache() 
                                                         
parser = argparse.ArgumentParser(description='inference')
parser.add_argument('--path', type=str, help='path to dataset')
parser.add_argument('--modelpath', type=str, help='path of model')
parser.add_argument('--outpath', type=str, help='path of heatmap output')
parser.add_argument('--device', type=int, help='gpu device')
parser.add_argument('--fold', type=int, default=5, help='fold for CV (default: 5)')
parser.add_argument('--model', type=str, help='name of model')
parser.add_argument('--method', type=str, help='LRP, IG or GGC')
parser.add_argument('--batch', default=1, type=int, help=' batch size for heatmap (default: 1)')
args = parser.parse_args()

## for GGC only
if args.model.startswith('modelB'):
	l_number = 16
elif args.model.startswith('modelA'):
	l_number = 16
elif args.model.startswith('modelD'):
	l_number = 8
elif args.model.startswith('modelC'):
	l_number = 12


dataset_ori  = DatasetFromNii_aug(csv_path = args.path, minmax=True)
model_pth = args.modelpath
out_pth = args.outpath
pathlib.Path(out_pth).mkdir(parents=True, exist_ok=True)
print(out_pth)


##### calculate and save heatmap for each fold
for i in range(0, args.fold): 
	torch.cuda.empty_cache()
	print(i)
	model=get_model(args.model, args.device)
	model.load_state_dict(torch.load(model_pth + "state_dict_f" + str(i) + ".pth"))
	model = model.cuda(args.device)

	if args.method == 'LRP':
		inn_model = InnvestigateModel(model, beta = 0.5, method = "b-rule").cuda(args.device)
	elif args.method == 'GGC':
		guided_gc = GuidedGradCam(model, layer = model.features[l_number])

	t,v,test = dataset_cv(i, args.fold, dataset_ori)
	data_loader = DataLoader(test, batch_size = args.batch, num_workers = 0, shuffle = False)

	all_prediction = []
	all_target = []
	all_attribution = []

	for j, (data, target) in enumerate(data_loader):  
		model.eval()
		torch.cuda.empty_cache()
		all_target += target.tolist()   
		d = data.cuda(args.device)
		d = Variable(d)

		output = model(d)
		_, prediction = torch.max(output.data, 1)
		prediction = prediction.cpu().detach().numpy()
		all_prediction += prediction.tolist()

		if args.method == 'LRP':
			_, attribution = inn_model.innvestigate(in_tensor=d, rel_for_class=target)
		elif args.method == 'IG':
			name = 'z' # zero as basline
			step = 100 # steps for IG
			attributions = integrated_gradients(d, model, target, name, step, cuda=True, args.device)
		elif args.method == 'GGC':
			attribution = guided_gc.attribute(d, target = target, interpolate_mode = "trilinear")
		
		all_attribution.append(attribution.cpu().detach().numpy())


	print(f"all_prediction:{all_prediction}")
	print(f"all_target: {all_target}")

	np.save(out_pth + "/all_prediction_f" + str(i) + ".npy", all_prediction)
	np.save(out_pth + "/all_target_f" + str(i) + ".npy", all_target)
	np.save(out_pth + "/all_attribution_f" + str(i) + ".npy", all_attribution)


##### calculate average heatmap from five fold
img_affine = np.array([[   1.,    0.,    0.,  96.],
					   [   0.,    1.,    0., 114.],
					   [   0.,    0.,    1.,  -96.],
					   [   0.,    0.,    0.,    1.]])

for i in range(0,5):
	all_target = np.load(out_pth + "/all_target_f" + str(i) + ".npy")
	all_prediction = np.load(out_pth + "/all_prediction_f" + str(i) + ".npy")
	all_attribution = np.load(out_pth + "/all_attribution_f" + str(i) + ".npy")
	all_attribution = np.squeeze(all_attribution, 1)
	av_ALL = get_ave_data(all_target ,all_prediction, all_attribution)
	new_image = nib.Nifti1Image(av_ALL, affine = img_affine)
	nib.save(new_image, out_pth + "map_f" + str(i) + ".nii")
    
map_all = np.zeros([193, 229, 193])
for i in range(0,5):
	tm = nib.load(out_pth + "map_f" + str(i) + ".nii")
	tmm = tm.get_fdata()
	map_all = tmm + map_all

map_av = map_all / 5
img = nib.Nifti1Image(map_av,img_affine)
nib.save(img, out_pth + "map_all.nii")
















