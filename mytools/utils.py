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
import random
from scipy.ndimage import rotate, zoom
from .models import *
from sklearn.preprocessing import MinMaxScaler

torch.cuda.empty_cache() 

def get_model(which_model,device=0):
	if which_model == "modelA24":
		model = modelA24(24).cuda(device)
	elif which_model == "modelA28":
		model = modelA28(28).cuda(device)
	elif which_model == "modelA32":
		model = modelA32(32).cuda(device)
	elif which_model == "modelA36":
		model = modelA36(36).cuda(device)
	elif which_model == "modelA40":
		model = modelA40(40).cuda(device)
	elif which_model == "modelA44":
		model = modelA44(44).cuda(device)
	elif which_model == "modelA48":
		model = modelA48(48).cuda(device)
	elif which_model == "modelA52":
		model = modelA52(52).cuda(device)


	elif which_model == "modelB24":
		model = modelB24(24).cuda(device)
	elif which_model == "modelB28":
		model = modelB28(28).cuda(device)
	elif which_model == "modelB32":
		model = modelB32(32).cuda(device)
	elif which_model == "modelB36":
		model = modelB36(36).cuda(device)
	elif which_model == "modelB40":
		model = modelB40(40).cuda(device)
	elif which_model == "modelB44":
		model = modelB44(44).cuda(device)
	elif which_model == "modelB48":
		model = modelB48(48).cuda(device)
	elif which_model == "modelB52":
		model = modelB52(52).cuda(device)

	elif which_model == "modelC24":
		model = modelC24(24).cuda(device)
	elif which_model == "modelC28":
		model = modelC28(28).cuda(device)
	elif which_model == "modelC32":
		model = modelC32(32).cuda(device)
	elif which_model == "modelC36":
		model = modelC36(36).cuda(device)
	elif which_model == "modelC40":
		model = modelC40(40).cuda(device)
	elif which_model == "modelC44":
		model = modelC44(44).cuda(device)
	elif which_model == "modelC48":
		model = modelC48(48).cuda(device)
	elif which_model == "modelC52":
		model = modelC52(52).cuda(device)

	elif which_model == "modelD24":
		model = modelD24(24).cuda(device)
	elif which_model == "modelD28":
		model = modelD28(28).cuda(device)
	elif which_model == "modelD32":
		model = modelD32(32).cuda(device)
	elif which_model == "modelD36":
		model = modelD36(36).cuda(device)
	elif which_model == "modelD40":
		model = modelD40(40).cuda(device)
	elif which_model == "modelD44":
		model = modelD44(44).cuda(device)
	elif which_model == "modelD48":
		model = modelD48(48).cuda(device)
	elif which_model == "modelD52":
		model = modelD52(52).cuda(device)

	return model


def get_ave_data(all_target ,all_prediction, all_attributions):
    outputs = []
    for j in range(len(all_target)):
        if (all_prediction[j] == 1) and (all_target[j] == 1):
            output = "TP"
        if (all_prediction[j] == 0) and (all_target[j] == 0):
            output = "TN"
        if (all_prediction[j] == 0) and (all_target[j] == 1):
            output = "FN"
        if (all_prediction[j] == 1) and (all_target[j] == 0):
            output = "FP"
        outputs += [output]

    TP_temp = outputs.count('TP')
    print(f"TP: {TP_temp}")
    TN_temp = outputs.count('TN')
    print(f"TN: {TN_temp}")
    FP_temp = outputs.count('FP')
    print(f"FP: {FP_temp}")
    FN_temp = outputs.count('FN')
    print(f"FN: {FN_temp}")

    _,_,a,b,c = all_attributions.shape

    temp = np.zeros((1,a,b,c))
    for i in range(len(all_attributions)):
        temp = temp + all_attributions[i]
    all_ = np.squeeze(temp)
    all_ave = all_/len(all_attributions)

    return all_ave




def train_earlystop(train_loader, valid_loader, model, out_pth, device, optimizer, loss_func, patience, n_epochs, delta, i):
        
    train_losses = []       # temporary store training loss for every epoch    
    valid_losses = []       # temporary store validation loss for every epoch
    avg_train_losses = []   # store average training loss for all epochs
    avg_valid_losses = []   # store average validation loss for all epochs 
    
    counter = 0             # initiate for early stop
    val_loss_min = None     # initiate for early stop
    early_stop = False      # initiate for early stop
    
    for epoch in range(1, n_epochs + 1):
        
        model.train()  
        
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.cuda(device), target.cuda(device)
            data, target = Variable(data), Variable(target) 
            optimizer.zero_grad()   
            output = model(data)            
            loss = loss_func(output, target)
            loss.backward()            
            optimizer.step()
            train_losses.append(loss.item())
  
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.cuda(device), target.cuda(device)
                data, target = Variable(data), Variable(target)
                output = model(data)          
                loss = loss_func(output, target)
                valid_losses.append(loss.item())

        
        train_loss = np.average(train_losses)  
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' + f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        train_losses = []   # clear lists to track next epoch
        valid_losses = []
     
        # check early stop

        if val_loss_min == None:
            val_loss_min = valid_loss
        elif valid_loss < val_loss_min - delta:
            val_loss_min = valid_loss
            print(f'valid loss is decreasing, saving model, reset count to zero')
            counter = 0
            torch.save(model.state_dict(), pth + 'checkpoint_' + str(i) +'.pt') 
        elif valid_loss >= val_loss_min - delta :
            counter +=1
            print(f'valid loss is not decreasing, start to count with : {counter} out of {patience}')
            if counter >= patience:
                print('early stop')
                break
    model.load_state_dict(torch.load(out_pth + 'checkpoint_' + str(i) +'.pt'))
    return  model, avg_train_losses, avg_valid_losses



class DatasetFromNii_aug(Dataset):    
    def __init__(self, csv_path, minmax = None, flip = None, shift = None, zoom = None, rotate = None, gass = None):

        # Transforms
        self.to_tensor = transforms.ToTensor()
        # read cvs
        self.data_info = pd.read_csv(csv_path, header=None)
        # get arrary of image path
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # get arrary of label
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])

        self.data_len = len(self.data_info.index)
        self.flip = flip
        self.shift = shift
        self.zoom = zoom
        self.rotate = rotate
        self.gass = gass
        self.minmax = minmax

    def __getitem__(self, index):
        # read single nii
        single_image_path = self.image_arr[index]
        single_image_nii = nib.load(single_image_path)
        single_image_arrary = single_image_nii.get_fdata()
        single_image_arrary=single_image_arrary.astype(np.float32)
        
        if self.minmax == True:
            single_image_arrary = (single_image_arrary-np.min(single_image_arrary))/(np.max(single_image_arrary)-np.min(single_image_arrary))

        #### flip ###
        flip_chance = random.random()
        if self.flip == 0 and flip_chance > 0.5:
            single_image_arrary = np.flip(single_image_arrary, axis = 0).copy()
        else:
            single_image_arrary = single_image_arrary
            
        #### gass ###
        #gass_chance = random.random()
        if self.gass:
            mask = nib.load("/home/wangd2/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii") 
            mask = mask.get_fdata()
            noise = np.random.normal(0, self.gass, (193, 229, 193))
            noise_mask = np.multiply(noise, mask)
            noise_mask = np.flip(noise_mask,1)
            single_image_arrary = single_image_arrary + noise_mask
            
        #### shift ###
        shift_step = rand([0], -20, 20)   # [0]: no 0
        shift_step_2 = rand([0], -20, 20)
        l1 = [0,1,2]
        shift_axis1 = random.choice(l1)
        l1.remove(shift_axis1)
        l2 = l1
        shift_axis2 = random.choice(l2)
        
        if self.shift:
            single_image_arrary = np.roll(single_image_arrary, shift_step, axis = shift_axis1).copy()
            single_image_arrary = np.roll(single_image_arrary, shift_step_2, axis = shift_axis2).copy()
            
        #### zoom ###  
        zoom_ratio = random.choice([1.1, 1.2, 1.3, 1.4]) 
        if self.zoom:
            single_image_arrary = zoom(single_image_arrary, zoom_ratio, order=1, prefilter=False).copy() 
            a,b,c = single_image_arrary.shape
            a1,b1,c1 = a//2,b//2,c//2
            single_image_arrary = single_image_arrary[a1-96:a1+97, b1-114:b1+115, c1-96:c1+97]
            
        #### rotate ### 
        rotate_angle = rand([0], -15, 15)
        if self.rotate:
            single_image_arrary = rotate(single_image_arrary, angle = rotate_angle, reshape = False).copy() 
            
        #add dimension for 3d conv
        single_image_arrary = np.expand_dims(single_image_arrary, axis=0)
        single_image_arrary=single_image_arrary.astype(np.float32)
        
        # to tensor
        img_as_tensor = torch.from_numpy(single_image_arrary)

        # get label
        single_image_label = self.label_arr[index]
        label_temp = np.array(single_image_label)
        label_as_tensor = torch.from_numpy(label_temp)
        return (img_as_tensor, label_as_tensor)
 
    def __len__(self):
        return self.data_len




def dataset_cv(i, k, dataset):
	fold_size = len(dataset) // k
	idx = list(range(len(dataset)))  
	if i == 4:
		Test_idx = idx[i* fold_size :i * fold_size + fold_size + 2]
		TrainVal_idx = idx[0 :i * fold_size]
		TrainVal_set = data.Subset(dataset, TrainVal_idx)
		lengths = [322,78]
		train, val = torch.utils.data.random_split(TrainVal_set, lengths)
		test = data.Subset(dataset, Test_idx)
	else:
		Test_idx = idx[i* fold_size :i * fold_size + fold_size]
		test = data.Subset(dataset, Test_idx)
		TrainVal_idx1 = idx[0 :i * fold_size]
		TrainVal_set1 = data.Subset(dataset, TrainVal_idx1)
		TrainVal_idx2 = idx[i * fold_size + fold_size :len(dataset)]
		TrainVal_set2 = data.Subset(dataset, TrainVal_idx2)
		TrainVal_set = torch.utils.data.ConcatDataset([TrainVal_set1, TrainVal_set2])
		lengths = [322,80]
		train, val = torch.utils.data.random_split(TrainVal_set, lengths)

	return train, val, test




