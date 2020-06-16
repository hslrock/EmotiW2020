import torch
from torch import nn
from module import resnet_o
import torch.nn.functional as F

class Face_Feature(nn.Module):
    
    def __init__(self,resnet):
        super(Face_Feature, self).__init__()
        self.resnet=resnet
        self.fc1=nn.Linear(1000,1)

        self.tanh=nn.Tanh()
    def forward(self,x):
  
        x=self.resnet(x)  
        x1=self.tanh(self.fc1(x))
        return x1
    
