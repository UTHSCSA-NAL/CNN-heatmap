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
import pathlib
import sys
torch.cuda.empty_cache() 

def integrated_gradients(input_, model, target_class, baseline_name, steps, cuda=True, device):
	input_size = input_.shape
	
	if baseline_name == "z":                                           # zero baseline
		baseline = np.zeros(input_size)
	elif baseline_name == "u":                                         # uniform noise baseline
		baseline = torch.rand(input_size)                    
	elif baseline_name == "g":                                         # add gaussian noise baseline
		baseline = input_ + torch.randn(input_size)       
		
	i_images = [ baseline + (i / steps) *(input_ - baseline) for i in range(0, steps + 1)] 
 
	all_grad = []
	for img in i_images:
		img = torch.tensor(img, dtype=torch.float32, device=device, requires_grad=True)
		out = model(img)  
		model.zero_grad()
		target_out = out[:, target_class.item()]
		target_out.backward()
		gradient = img.grad.detach().cpu().numpy()[0]
		all_grad.append(gradient)
		
	grad = np.array(all_grad)
	av_grad = np.average(grad[:-1], axis=0)   # average image gradient
	diff = (input_ - baseline).detach().squeeze(0).cpu().numpy()
	i_grads = diff * av_grad
	
	return i_grads
