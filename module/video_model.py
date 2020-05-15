import torch
from torch import nn
import torch.nn.functional as F
from module import transformer,densenet
from src import detect_faces, show_bboxes

class Video_Feature(nn.Module):
    
    def __init__(self,frame=24,dim=1000,joint=True,grayscale=True,classfication=True,skip_process=True):
        super(Video_Feature, self).__init__()
        self.label=1
        self.classfication=classfication
        self.skip_process=skip_process
        if classfication:
            self.label=3

        #self.resnet_1=resnet.resnet18(dim=dim,grayscale=grayscale)
        
        self.densenet=densenet.densenet121(pretrained=True)
        self.dim_frame=1000
        self.dim_face=100
        #self.fc1=nn.Linear(dim,1024)
        self.fc2=nn.Linear(1000,512)
        self.fc3=nn.Linear(512, self.label)
        self.joint=joint
        self.posencoding=transformer.PositionalEncoding(1000,n_position=frame)
        
        self.transformer=transformer.MultiHeadAttention(d_k=256,d_v=256,n_head=2,frame=frame)
        
    def frame_embedder(self,x,t):
        x1 = (self.densenet(x[:, t, :, :, :]))  # Pretrained_Densenet
        x1 = x1.view(x1.size(0), -1)
        
        return x1
    def check_raw(self,x,dim):
        if x.shape[-1]==dim:
            return False
        return True 
        
        return None
    def stack_frame(self,x):
        ##Transformation to: Frames*Channel*width*height

        cnn_embed_seq = []
     
        for t in range(x.size(1)):
            with torch.no_grad():
                x1=self.frame_embedder(x,t)
                cnn_embed_seq.append(x1)            
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        
        return cnn_embed_seq
        
    def forward(self,x):
        if  self.check_raw(x,self.dim_frame):
            print("hi")
            cnn_embed_seq=self.stack_frame(x)  
            cnn_embed_seq=self.posencoding(cnn_embed_seq) 
           # return cnn_embed_seq
        else:
            cnn_embed_seq=x
        ##Input to Transformer
        
        cnn_embed_seq=self.fc2(cnn_embed_seq)
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

class Encoder(nn.Module):
    
    def __init__(self,num_frame=24,dim=1000,model=None,isframe=0):
        super(Encoder, self).__init__()
        self.label=3
        if model is None:
            self.embedder=densenet.densenet121(pretrained=False)
            self.fc1=nn.Linear(1000,512)
        else:
            self.embedder=model
        if isframe==1:
            self.fc1=nn.Linear(1000,256)
        else:
            self.fc1=nn.Linear(100,256)
        self.dim_frame=dim
        self.num_frame=num_frame
        self.fc1=nn.Linear(1000,512)
        self.fc2=nn.Linear(512, self.label)
        
        self.posencoding=transformer.PositionalEncoding(self.dim_frame,n_position=num_frame)
        self.transformer=transformer.MultiHeadAttention(d_k=128,d_v=128,n_head=2,frame=num_frame)
        
    def img_embedder(self,x,t,embedder):
        x1 = (embedder(x[:, t, :, :, :]))  # Pretrained_Densenet
        x1 = x1.view(x1.size(0), -1)
        
        return x1
    
    
    def check_raw(self,x,dim):
        if x.shape[-1]==dim:
            return False
        return True 
        
        return None
    def stack_frame(self,x):
        ##Transformation to: Frames*Channel*width*height

        cnn_embed_seq = []
     
        for t in range(x.size(1)):
            with torch.no_grad():
                x1=self.img_embedder(x,t,self.embedder)
                cnn_embed_seq.append(x1)            
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        
        return cnn_embed_seq
        
    def forward(self,x):
        if  self.check_raw(x,self.dim_frame):
            cnn_embed_seq=self.stack_frame(x)  
            cnn_embed_seq=self.posencoding(cnn_embed_seq) 
           # return cnn_embed_seq
        else:
            cnn_embed_seq=x
        ##Input to Transformer
        
        cnn_embed_seq=self.fc1(cnn_embed_seq) 
        ##Dimension Reduction
        output=self.transformer(cnn_embed_seq,cnn_embed_seq,cnn_embed_seq)
        
        output,attention=output[0],output[1]
    
        
        output=(output.transpose(1,2))
        output=self.fc2(F.relu(output))
        #output=F.relu(output)
        output=output.squeeze(1)
        return F.softmax(output, 1)
