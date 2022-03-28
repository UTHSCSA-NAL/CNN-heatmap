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
from mytools.utils import *
import pathlib
import sys
import argparse
torch.cuda.empty_cache() 

parser = argparse.ArgumentParser(description='inference')
parser.add_argument('--path', type=str, help='path to dataset')
parser.add_argument('--modelpath', type=str, help='path of model to infer')
parser.add_argument('device', type=int, help='gpu device')
parser.add_argument('--fold', type=int, default=5, help='fold for CV (default: 5)')
parser.add_argument('--model', type=str, help='name of model')
parser.add_argument('--batch', default=5, type=int, help='batch size (default: 5)')

dataset_ori  = DatasetFromNii_aug(csv_path=args.path, minmax=True)

def inference(pth, dataset, which_model, device, k):
    for i in range(0,k):    
        _,_, Test_set = dataset_cv(i,k,dataset)
        
        Test_loader = DataLoader(Test_set, batch_size = args.batch, num_workers=0, shuffle = False)

        model = get_model(which_model,device)
        model.load_state_dict(torch.load(pth + "state_dict_f" + str(i) +".pth"))
        model = model.cuda(device)
        sig = nn.Softmax(dim=1)
        all_predict = []
        all_target = []
        all_probs = []
        
        model.eval()
        with torch.no_grad():
            for j, (data, target) in enumerate(Test_loader): 
                torch.cuda.empty_cache() 
                d,t = data.cuda(device), target.cuda(device)
                d,t = Variable(d), Variable(t)
                all_target += t.tolist()  
                outcome = model(d)
                ouput = sig(outcome)
                probs = ouput.cpu().detach().numpy() 
                #print(probs)
                all_probs.append(probs)  
                _, predicted = torch.max(ouput, 1)     
                predicted = predicted.cpu().detach().numpy()
                all_predict.append(predicted) 

        predict_pth = pth + "all_predict_f" + str(i) + ".npy"
        target_pth = pth + "all_target_f" + str(i) + ".npy"
        probs_pth = pth + "all_probs_f" + str(i) + ".npy"  
        np.save(probs_pth, all_probs) 
        np.save(predict_pth, all_predict) 
        np.save(target_pth, all_target) 


inference(args.modelpath, dataset_ori, args.model, args.device, args.fold)


def get_acc(all_target,all_prediction):
    outputs = []

    for i in range(len(all_target)):
        if (all_prediction[i] == 1) and (all_target[i] == 1):
            output = "TP"
        if (all_prediction[i] == 0) and (all_target[i] == 0):
            output = "TN"    
        if (all_prediction[i] == 0) and (all_target[i] == 1):
            output = "FN" 
        if (all_prediction[i] == 1) and (all_target[i] == 0):
            output = "FP" 
        outputs += [output] 
    
    TP = outputs.count('TP')
    TN = outputs.count('TN')
    FP = outputs.count('FP')
    FN = outputs.count('FN')
    acc = (TP + TN)/len(all_target)
    return acc, TP, TN, FP, FN
    
def all_fold(pth):

    acc={'fold':[], 'acc':[], 'TP':[], 'TN': [], 'FP':[],'FN':[]}
    
    for i in folds:
    
        all_prediction = np.load(pth + "all_predict_f"+ str(i) + ".npy", allow_pickle=True)
        all_target = np.load(pth + "all_target_f"+ str(i) + ".npy", allow_pickle=True)
        
        acc_tmp, TP_temp, TN_temp, FP_temp, FN_temp = get_acc(all_target,all_prediction)
        acc['fold'].append(i)
        acc['acc'].append(acc_tmp)
        acc['TP'].append(TP_temp)
        acc['TN'].append(TN_temp)
        acc['FP'].append(FP_temp)
        acc['FN'].append(FN_temp)
    
    #print(acc)
    acc_list = acc['acc']
    av_acc = np.average(acc['acc'])
    return acc, av_acc

#print("################ done #####################")
acc, av_acc = all_fold(args.modelpath)
av_acc =np.round(av_acc*100,2) 
print(f"average ACC: {av_acc}")
print(acc)
#print(np.array(acc['acc']).std())
#print("TP, TN, FP, FN", sum(acc['TP']), sum(acc['TN']), sum(acc['FP']), sum(acc['FN']))

