import torch
from torch import nn
from module import resnet
import torch.nn.functional as F
from module import transformer,densenet

class Video_Feature(nn.Module):
    
    def __init__(self,frame=16,dim=1000,joint=True,grayscale=True,classfication=True):
        super(Video_Feature, self).__init__()
        self.label=1
        self.classfication=classfication
        if classfication:
            self.label=3

        #self.resnet_1=resnet.resnet18(dim=dim,grayscale=grayscale)
        self.densenet=densenet.densenet121(pretrained=True)

        #self.fc1=nn.Linear(dim,1024)
        self.fc2=nn.Linear(1000,512)
        self.fc3=nn.Linear(512, self.label)
        self.joint=joint
        self.posencoding=transformer.PositionalEncoding(512)
        
        self.transformer=transformer.MultiHeadAttention(frame=frame)
    def forward(self,x):
  
        first=True
        cnn_embed_seq = []
        
        ##Transformation to: Frames*Channel*width*height
        
        for t in range(x.size(1)):
            with torch.no_grad():

                x1 = self.fc2(self.densenet(x[:, t, :, :, :]))  # ResNet
                #print(x[:, t, :, :, :].shape)
                x1 = x1.view(x1.size(0), -1)
               # print(x[:, t, :, :, :].shape)
                cnn_embed_seq.append(x1)
              
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        cnn_embed_seq=self.posencoding(cnn_embed_seq) 
        
        
        ##Input to Transformer
        
        output=self.transformer(cnn_embed_seq,cnn_embed_seq,cnn_embed_seq)
        output,attention=output[0],output[1]
        #print(attention.shape)
       # output=self.fc2(F.relu((self.fc1(output.transpose(1,2)))))
        output=(output.transpose(1,2))
        if self.joint==False:
            output=self.fc3(F.relu(output))
        
        
        if self.classfication:
            output=F.relu(output)
            output=output.squeeze(1)
            if self.joint==True:
                return output
            else:
                return F.softmax(output, 1)
        else:
            return output.squeeze(1)
        
        
def num_correct(prediction,labels):
    correct=0
    for i,(pred_label,label) in enumerate(zip(prediction,labels)):
        if (pred_label.item()==label.item()):
            correct +=1
    return correct        