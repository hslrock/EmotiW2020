import torch
from torch import nn
from module import resnet_o
import torch.nn.functional as F

class Face_Feature(nn.Module):
    
    def __init__(self):
        super(Face_Feature, self).__init__()
        self.resnet1=resnet_o.resnet18()
        self.fc1=nn.Linear(100,1)
        self.tanh=nn.Tanh()
    def forward(self,x):
  
        x=self.resnet1(x)  
        x1=self.tanh(self.fc1(x))
        return x1
    
if __name__ == '__main__':
    print("hi")
