import torch
from torch import nn
import torch.nn.functional as F
from module import transformer,densenet
from src import detect_faces, show_bboxes
import numpy as np
class Encoder(nn.Module):
    
    def __init__(self,num_frame=24,dim=1000,model=None,isframe=0,label=3):
        super(Encoder, self).__init__()
        self.label=label
        if model is None:
            self.embedder=densenet.densenet121(pretrained=True)
            self.fc1=nn.Linear(1000,128)
        else:
            self.embedder=model
            self.fc1=nn.Linear(1000,128)
        self.isframe=isframe
        self.dim_frame=dim
        self.num_frame=num_frame


        self.fc2=nn.Linear(128, self.label)
        
        self.posencoding=transformer.PositionalEncoding(self.dim_frame,n_position=num_frame)
        self.transformer=transformer.MultiHeadAttention(d_model=128,d_k=64,d_v=64,n_head=2,frame=num_frame)
        
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

        cnn_embed_seq=x
        cnn_embed_seq=self.posencoding(cnn_embed_seq) 
        ##Dimension Reduction
        output=self.transformer(cnn_embed_seq,cnn_embed_seq,cnn_embed_seq)
        
        output,attention=output[0],output[1]
    
        
        output=(output.transpose(1,2))
        
        
        if self.isframe ==0:
            return output
        
        #output=F.relu(output)
        output=F.relu(output)
        output=self.fc2(output)
        output=output.squeeze(1)
        return output
    
class Audeo_modeller(nn.Module):
    def __init__(self,softmax=True,label=3):
        super(AudioRecognition, self).__init__()
        self.label=label
        self.fc1=nn.Linear(988,512)
        self.fc2=nn.Linear(512,128)
        self.fc3=nn.Linear(128,self.label)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.bn2=nn.BatchNorm1d(num_features=128)
        self.relu = nn.ReLU()
        self.softmax=softmax
        
    def forward(self,x):
        x=self.relu(self.bn1(self.fc1(x)))
        x=self.relu(self.bn2((self.fc2(x))))
        x=(self.fc3(x))
        
        if self.softmax:
            return F.softmax(x,1)
        else:
            return self.relu(x)
class Video_modeller(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self,frame,face_model,frame_model,softmax=True,pre_train=True):
        super().__init__()
        self.num_frame=frame
        self.num_face=5
        
        
        
        self.frame_model=frame_model
        self.face_model=face_model
        
        
        self.en1=Encoder(num_frame=5,dim=128,model=face_model,isframe=0)
        self.en2=Encoder(num_frame=frame,dim=128,model=None,isframe=pre_train)
        
        self.fc1=nn.Linear(1000,128)
        self.fc2=nn.Linear(1000,128)
        self.fc3=nn.Linear(256,128)
        
        
        self.bn1_frame=nn.BatchNorm1d(num_features=128)
        self.bn1_face=nn.BatchNorm1d(num_features=128)

        self.softmax=softmax
        
    def check_raw(self,x,dim):
        return x.shape[-1]==dim
    def frame_embedder(self,x,t,embedder):
        x1 = (embedder(x[:, t, :, :, :]))  # Pretrained_Densenet
        x1 = x1.view(x1.size(0), -1)
        
        return x1
        
    def face_embedder(self,x,t,encoder):
        x1 = (encoder(x[:, t,:]))  # Pretrained_Densenet            
        x1 = x1.view(x1.size(0), -1)
        return x1

    def stack_face_encoder(self,x):
        ##Transformation to: Frames*Channel*width*height
        
        if not self.check_raw(x,1000):
            frame_sequence = []
            for frame in range(x.size(1)):
                face_sequence=torch.empty(size=(self.num_face,x.size(0),1000))
                x1=x[:, frame, :,:, :]
                
                for face in range(x.size(2)):
                    x2=x1[:,face,:,:]
                    x2=self.face_model(x2)
                    face_sequence[face]=x2              
                face_sequence=face_sequence.transpose_(0,1)              
                frame_sequence.append(face_sequence)
            x=torch.stack(frame_sequence, dim=0)

            x=self.fc2(x)


        cnn_embed_seq = []
        for t in range(x.size(1)):       
            x1=self.face_embedder(x,t,self.en1)
            cnn_embed_seq.append(x1) 
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)

        return cnn_embed_seq
    
    def stack_frame(self,x):
        ##Transformation to: Frames*Channel*width*height
        frame_seq=torch.empty(size=(self.num_frame,x.size(0),1000))

        for t in range(x.size(1)):
            with torch.no_grad():
                x1=self.frame_embedder(x,t,self.embedder)
                frame_seq[t]=x1
        
        return frame_seq.transpose_(0, 1)
    
    def forward(self,x,y):
        if y is not None:
                frame_local=self.stack_face_encoder(y)

        if x is not None:
            frame_global=self.stack_frame(x)
            frame_global=self.fc2(frame_global)
            
        if y is None:
            
            return self.en2(frame_global)
        
        if x is None:
            return self.en2(frame_local)            
  
        
        frames_embedding=torch.cat((frame_global,frame_local),2)

        frames_embedding=self.fc3(frames_embedding)
        video_embed=self.en2(frames_embedding) 
        if self.softmax:
            return F.softmax(video_embed,1)
        else
            return video_embed

class Video_transformer(nn.Module):

    