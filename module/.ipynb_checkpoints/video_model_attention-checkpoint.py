import torch
from torch import nn
import torch.nn.functional as F
from module import transformer_attention as transformer
import numpy as np
import torchvision.models as models

class Encoder(nn.Module):
    
    def __init__(self,num_frame=24,dim=1000,model=None,joint=0,label=3,pos=True):
        super(Encoder, self).__init__()
        #Defines num of labels
        self.label=label 
        #Defines whether training is only on images or with audio
        self.joint=joint 
        #Defines number(max) of frames of video
        self.num_frame=num_frame 
        #Defines dimension of video vectory
        self.dim_frame=dim
        ##Encoder for Frame/Face
        self.pos=pos  
        
        #Transformer module is directly used from https://github.com/jadore801120/attention-is-all-you-need-pytorch
        self.fc1=nn.Linear(128, self.label) 
        self.posencoding=transformer.PositionalEncoding(dim,n_position=num_frame) # Add temporal information to each frames
        self.transformer=transformer.MultiHeadAttention(d_model=128,d_k=128,d_v=128,n_head=1,frame=num_frame) # Find self attnetion between the frames/faces
        
    def forward(self,x):
        
        #If we are training on encoding faces, they do not have sequential information, so positional encoding is not requried
        x=self.posencoding(x)  if self.pos else x
        ##Dimension Reduction
        output,attention,_=self.transformer(x,x,x)  
        
        # Jointing Training requires feature vector as output
        # if isframe==1, it would give output as label vecltor [positive,negative neutral]
        if self.joint ==0:
            return output.squeeze(1)
        
        output=F.relu(output)
        output=self.fc1(output)
        output=output.squeeze(1)
        
        return F.softmax(output, 1)
    
class video_transformer(nn.Module):
    #Model to concaneates the video embedding and audio embedding
    def __init__(self,video_model,audio_model,pre_train=True):
        super().__init__()
        self.video_model=video_model
        self.audio_model=audio_model
        self.fc1=nn.Linear(256,128)
        self.fc2=nn.Linear(128,3)
        self.relu=nn.ReLU()
   

    def forward(self,x,y,z):
        x=self.video_model(x,y)
        x=x.squeeze(1)
        z=self.audio_model(z)

        output=torch.cat((x,z),1)
        output=self.fc2(self.relu(self.fc1(output)))
        return F.softmax(output,1)
    

class Video_modeller(nn.Module):

    def __init__(self,frame,frame_model=None,face_model=None,pre_train=True):
        super().__init__()
        
        ##Num of Frames/face extracted
        self.num_frame=frame
        self.num_face=5
        
        #Loading Pretrained Indirectly/Directly
        self.frame_model=models.resnet18(pretrained=True) if frame_model==None else frame_model    
        self.face_model=models.resnet18(pretrained=True) if face_model==None else face_model
      
        #Load Transformer Encoder
            
        self.en1=Encoder(num_frame=5,dim=128,model=self.face_model,joint=0,pos=False)
        self.en2=Encoder(num_frame=frame,dim=128,model=self.frame_model,joint=pre_train)
        
        
        #Dimension Reduction from dim= 1000 frame/face feature (face_feature reduced to be later)
        self.fc1=nn.Linear(1000,128)
        self.fc2=nn.Linear(1000,128)
        self.fc3=nn.Linear(256,128)
        
        self.dropout=nn.Dropout(0.1)
    #Embeds face image(2D) to a vector (1D)
    #Embeds frame image(2D) to a vector (1D)
    def frame_embedder(self,x,t,embedder):
        x1 = (embedder(x[:, t, :]))  # Similar to above
        return x1    
    
    ##To be removed
    def check_raw(self,x,dim):
        return x.shape[-1]==dim

    def stack_face_encoder(self,x):
        #x=[batch_size:frame:face:channel:legth:width]
        if not self.check_raw(x,1000):
            frame_sequence = torch.empty(size=(x.size(1),x.size(0),x.size(2),1000))

            for frame in range(x.size(1)): #For every Frame
                face_sequence=torch.empty(size=(self.num_face,x.size(0),1000))
                x1=x[:, frame, :,:, :]
                
                for face in range(x.size(2)): #For every Face in the frame
                    with torch.no_grad():
                        x2=x1[:,face,:,:]
                        x2=self.face_model(x2) #x2 =[batchsize*1000]
                        face_sequence[face]=x2 
                
                
                face_sequence=face_sequence.transpose_(0,1)  #face_num*batch_num*1000->batch*face*1000

                
                frame_sequence[frame]=face_sequence #frame*batch*face_num*1000
            frame_sequence=frame_sequence.transpose_(0,1)

            x=self.fc2(frame_sequence) #x=batch*frame*face_num*128

        face_seq = torch.empty(size=(x.size(1),x.size(0),128))
        for t in range(x.size(1)):       
            x1=self.en1(x[:,t,:]) 
            face_seq[t]=x1
        return face_seq.transpose_(0, 1) #return=batch*frame*128
    
    def stack_frame(self,x):
        ##Transformation to: Frames*Channel*width*height
        frame_seq=torch.empty(size=(self.num_frame,x.size(0),1000))

        for t in range(x.size(1)):
            with torch.no_grad():
                x1=self.frame_model(x[:, t, :])
                frame_seq[t]=x1
        
        return frame_seq.transpose_(0, 1) #frame_seq=batch*frame*1000
    
    def forward(self,x,y):
        if y is not None:
            frame_local=self.stack_face_encoder(y)
                

        if x is not None:

            frame_global=self.stack_frame(x)  
            frame_global=self.fc2(frame_global) #frame_seq=batch*frame*128
            
        if y is None:
            
            return self.en2(frame_global)
        
        if x is None:
            return self.en2(frame_local)            
  
        
        frames_embedding=torch.cat((frame_global,frame_local),2) #global_embedding (concat) local_embedding

        frames_embedding=self.fc3(frames_embedding)
        video_embed=self.en2(frames_embedding) 

        return video_embed

class AudioRecognition(nn.Module):
    
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
        x=self.bn1(self.fc1(x))
        x=self.bn2(self.fc2(x))
        
        
        if self.softmax:
            x=(self.relu(self.fc3(x)))
            return F.softmax(x,1)
        else:
            return x
        
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