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
	if baseline_name == "z":
		baseline = 0 * input_
	elif baseline_name == "u":
		baseline = torch.rand(1, 1, 193, 229, 193)                 # [0, 1]   
	elif baseline_name == "g":
		baseline = inputs + torch.randn(1, 1, 193, 229, 193)       # gaussian
		
	i_images = [ baseline + (i / steps) *(input_ - baseline) for i in range(0, steps + 1)] 
 
	all_grad = []
	for img in i_images:
		img = torch.tensor(img, dtype=torch.float32, device=device, requires_grad=True)
		outputs = model(img)  
		model.zero_grad()
		outputs[:,target_class.item()].backward()
		gradient = img.grad.detach().cpu().numpy()[0]
		all_grad.append(gradient)
		
	grad = np.array(all_grad)
	av_grad = np.average(grad[:-1], axis=0)   # average image gradient
	diff = (input_ - baseline).detach().squeeze(0).cpu().numpy()
	i_grads = diff * av_grad
	
	return i_grads

