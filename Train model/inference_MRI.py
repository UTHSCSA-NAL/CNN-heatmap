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
from mytools.models_MRI import ClassificationModel3D_mri0,ClassificationModel3D_mri1, ClassificationModel3D_mri7, ClassificationModel3D_mri4
from mytools.models_cut import ClassificationModel3D_mri_cut, ClassificationModel3D_mri_cut1, ClassificationModel3D_mri_cut2
from mytools.models import DatasetFromNii
from mytools.models_res import MyResNet,MyResNet1,MyResNet2, MyResNet0
from mytools.my3DCNN         import  divNet_L4, divNet_L4_leaky
from mytools.my3DCNN_resized import  divNet_rL4, divNet_rL4_leaky, divNet_rL3, divNet_rL5, conNet_rL4, divNet_rL6, divvgg_rL8,res_rL18,ddvNet_rL5

torch.cuda.empty_cache() 

#mod = "S3"
#dataset_MRI  = DatasetFromNii(csv_path='data/t1_mean_'+ mod +'.csv')
#out_pth = "/home/wangd2/LRP/Train/MRI_experiment/smooth/" + mod+ "/"


out_pth = "/home/wangd2/LRP/Train/MRI_experiment/aug/model0_gall/"
dataset_MRI = DatasetFromNii(csv_path='data/t1_mean.csv')


print(out_pth)
b=1
k=5
def inference_mri(pth, dataset):

    for i in range(k):
        print(i)
        
        fold_size = len(dataset) // k
        idx = list(range(len(dataset))) # idx = [0, 1, 2,....500]
        Test_idx = idx[i* fold_size :i * fold_size + fold_size]  
        Test_set = Subset(dataset, Test_idx)
        Test_loader = DataLoader(Test_set, batch_size=b, num_workers=0, shuffle = False)
        
        device = 0
        model = ClassificationModel3D_mri0().cuda(device)
        model.load_state_dict(torch.load(pth + "state_dict_f" + str(i) +".pth"))
        model = model.cuda(device)

        
        sig = nn.Sigmoid()
        all_predict = []
        all_target = []
        all_probs = []
        
        model.eval()
        with torch.no_grad():
            for j, (data, target) in enumerate(Test_loader): 
             
                torch.cuda.empty_cache() 
                d,t = data.cuda(device), target.cuda(device)
                d,t = Variable(d), Variable(t)
                #print(f"target: {t}")
                
                all_target += t.tolist()  # tensor to list
                
                outcome = model(d)
                _, predicted = torch.max(outcome.data, 1)     # same results as
                predicted = predicted.cpu().detach().numpy()
                all_predict.append(predicted)  # int can append
                
                ouput = sig(outcome)
                
                probs = ouput.cpu().detach().numpy()  
                     
                all_probs.append(probs)  # int can append  
    
        predict_pth = pth + "all_predict_f" + str(i) + ".npy"
        target_pth = pth + "all_target_f" + str(i) + ".npy"
        probs_pth = pth + "all_probs_f" + str(i) + ".npy"  
        
        np.save(probs_pth, all_probs) 
        np.save(predict_pth, all_predict) 
        np.save(target_pth, all_target) 

inference_mri(out_pth, dataset_MRI)
print("done inference MRI")



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

    for i in range(k):

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
#    best_fold = acc_list.index(max(acc['acc']))

#    return av_acc, best_fold, acc['TP'][best_fold], acc['TN'][best_fold], acc['FP'][best_fold], acc['FN'][best_fold]
    return acc, av_acc

print("################ MRI #####################")    
acc, av_acc = all_fold(out_pth)
print(f"average ACC: {av_acc}")
print(acc)
print("TP, TN, FP, FN", sum(acc['TP']), sum(acc['TN']), sum(acc['FP']), sum(acc['FN']))
#print(f"best fold: {a0}")
#print(f"TP: {b0}, TN: {c0}, FP: {d0}, FN: {e0}")

















    
    
