import torch
from torch import nn,optim

from module import resnet
from module import invres


class resampling(nn.Module):
    def __init__(self,dim=64,output_size=64):
        super(resampling, self).__init__()
        
        self.encoder=resnet.resnet18(dim=dim)
        self.decoder=invres.invresnet18(dim=dim,output_size=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.sig=nn.Sigmoid()
        self.os=output_size
    def forward(self,x):
        x=self.decoder(self.encoder(x))
        if self.os==32:
            x=self.maxpool(x)
        if self.os==16:
            x=self.maxpool(maxpool(x))
        return self.sig(x)


class attention_module(nn.Module):
    
    def __init__(self,output_size=64):
        super(attention_module,self).__init__()
        self.module=resampling(64,output_size)
        self.relu=nn.ReLU()
        self.output_size=output_size
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        
    def forward(self,x):
        
        x1=self.module(x)
        if self.output_size==32:
            x=self.maxpool(x)
        x2=x*x1        
        x2=x2+x
        return self.relu(x2)
        
        
class Attention(nn.Module):
    
    def __init__(self,output=[64,64,32],num_of_classes=10):
        super(Attention,self).__init__()
        self.attn1=attention_module(output[0])
        self.attn2=attention_module(output[1])
        self.attn3=attention_module(output[2])
        self.classification=resnet.resnet18(dim=num_of_classes)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.attn1(x)
        x=self.attn2(x)        
        x=self.attn3(x)
        x=self.classification(x)
        return self.relu(x)        