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
	 
	# ([51，1, 1, 193, 229, 193])
	interpolated_inputs = [baseline + (i / steps) *(input_ - baseline) for i in range(0, steps + 1)] 
   
	gradients = []
	for img in interpolated_inputs:
		
		#img = torch.Size([1, 193, 229, 193])
		img = torch.tensor(img, dtype=torch.float32, device=device, requires_grad=True)
		outputs = model(img)  # get output of img ex: [0.0002, 0.9068]
		
		model.zero_grad()
		outputs[:,target_class.item()].backward()
		gradient = img.grad.detach().cpu().numpy()[0]
		gradients.append(gradient)
		
	gradients = np.array(gradients)
	avg_grads = np.average(gradients[:-1], axis=0)                            # average image gradient

	diff_input = (input_ - baseline).detach().squeeze(0).cpu().numpy()
	integrated_grad = diff_input * avg_grads
	
	return integrated_grad

