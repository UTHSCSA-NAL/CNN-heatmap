import torch
import torch.nn as nn
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
import torchvision.models as models

################################
############# A ################
################################
class modelA24(nn.Module): 
    def __init__(self, n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(864, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                       
        return x


class modelA28(nn.Module): 
    def __init__(self, n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(1008, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                       
        return x

class modelA32(nn.Module): 
    def __init__(self, n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(1152, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                       
        return x

class modelA36(nn.Module): 
    def __init__(self, n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(1296, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                       
        return x

class modelA40(nn.Module): 
    def __init__(self, n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(1440, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                       
        return x

class modelA44(nn.Module):
    def __init__(self, n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(1584, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class modelA48(nn.Module): 
    def __init__(self, n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(1728, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                       
        return x

class modelA52(nn.Module): 
    def __init__(self, n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(1872, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                       
        return x


################################
############# B ################
################################
class modelB24(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 3),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(1920, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                          
        return x

class modelB28(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 3),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(2240, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                          
        return x

class modelB32(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 3),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(2560, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                          
        return x

class modelB36(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 3),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(2880, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                          
        return x

class modelB40(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 3),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(3200, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                          
        return x

class modelB44(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 3),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(3520, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                          
        return x

class modelB48(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 3),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(3840, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                          
        return x

class modelB52(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 3),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 3),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(4160, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                          
        return x


################################
############# C ################
################################
class modelC24(nn.Module): 
    def __init__(self,n):
        self.n=n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 5),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(1920, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                        
        return x

class modelC28(nn.Module): 
    def __init__(self,n):
        self.n=n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 5),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(2240, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                        
        return x

class modelC32(nn.Module):
    def __init__(self,n):
        self.n=n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 5),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(2560, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class modelC36(nn.Module): 
    def __init__(self,n):
        self.n=n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 5),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(2880, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                        
        return x


class modelC40(nn.Module): 
    def __init__(self,n):
        self.n=n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 5),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(3200, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                        
        return x

class modelC44(nn.Module): 
    def __init__(self,n):
        self.n=n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 5),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(3520, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                        
        return x

class modelC48(nn.Module):
    def __init__(self,n):
        self.n=n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 5),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(3840, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class modelC52(nn.Module): 
    def __init__(self,n):
        self.n=n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 5),     nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.n, self.n, 5),nn.BatchNorm3d(self.n),nn.ReLU(True), nn.MaxPool3d(2))
        self.classifier = nn.Sequential(nn.Linear(4160, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                        
        return x



################################
############# D ################
################################
class modelD24(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3))
        self.classifier = nn.Sequential(nn.Linear(1920, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                           
        return x

class modelD28(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3))
        self.classifier = nn.Sequential(nn.Linear(2240, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                           
        return x

class modelD32(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3))
        self.classifier = nn.Sequential(nn.Linear(2560, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                           
        return x

class modelD36(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3))
        self.classifier = nn.Sequential(nn.Linear(2880, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                           
        return x


class modelD40(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3))
        self.classifier = nn.Sequential(nn.Linear(3200, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                           
        return x

class modelD44(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3))
        self.classifier = nn.Sequential(nn.Linear(3520, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                           
        return x

class modelD48(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3))
        self.classifier = nn.Sequential(nn.Linear(3840, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                           
        return x

class modelD52(nn.Module): 
    def __init__(self,n):
        self.n = n
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv3d(1, self.n, 7),     nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3),
            nn.Conv3d(self.n, self.n, 7),nn.BatchNorm3d(self.n),nn.ReLU(True),nn.MaxPool3d(3))
        self.classifier = nn.Sequential(nn.Linear(4160, 64),nn.ReLU(True),nn.Dropout(),nn.Linear(64, 2))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)                                           
        return x






