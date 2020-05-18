import torch
from torch import nn
import torch.nn.functional as F
from module import transformer,densenet
from src import detect_faces, show_bboxes

class Encoder(nn.Module):
    
    def __init__(self,num_frame=24,dim=1000,model=None,isframe=0,label=3):
        super(Encoder, self).__init__()
        self.label=label
        if model is None:
            self.embedder=densenet.densenet121(pretrained=True)
            self.fc1=nn.Linear(1000,512)
        else:
            self.embedder=model
        self.isframe=isframe
        self.dim_frame=dim
        self.num_frame=num_frame
        self.fc1=nn.Linear(self.dim_frame,256)
        
        #self.fc1=nn.Linear(1000,512)
        self.fc2=nn.Linear(256, self.label)
        
        self.posencoding=transformer.PositionalEncoding(self.dim_frame,n_position=num_frame)
        self.transformer=transformer.MultiHeadAttention(d_model=256,d_k=128,d_v=128,n_head=2,frame=num_frame)
        
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
        output=F.relu(output)
        output=self.fc2(output)
        if self.isframe ==0:
            return output
        
        #output=F.relu(output)
        output=output.squeeze(1)
        return F.softmax(output, 1)
    
    
class Video_modeller(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self,frame):
        super().__init__()
        
        self.en1=Encoder(num_frame=5,dim=100,model=None,isframe=0,label=256)
        self.en2=Encoder(num_frame=frame,dim=256,model=None,isframe=1,label=3)
        
        self.fc1=nn.Linear(1000,256)
        self.fc2=nn.Linear(512,256)
    def face_embedder(self,x,t,encoder):
        x1 = (encoder(x[:, t,:]))  # Pretrained_Densenet            
        x1 = x1.view(x1.size(0), -1)
        
        return x1

    def check_raw(self,x,dim):
        if x.shape[-1]==dim:
            return False
        return True 
        
        return None
    def stack_face_encoder(self,x):
        ##Transformation to: Frames*Channel*width*height
        cnn_embed_seq = []
     
        for t in range(x.size(1)):
            with torch.no_grad():
                x1=self.face_embedder(x,t,self.en1)
                cnn_embed_seq.append(x1)            
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq
    
    def forward(self,x,y):
        initial_time=time.time()
        if y is not None:
            start=time.time()
            frame_local=self.stack_face_encoder(y)
            end=time.time()
            print(end-start)
        if x is not None:
            frame_global=self.fc1(x)
        if y is None:
            return self.en2(frame_global)
        if x is None:
            return self.en2(frame_local)            
            
        
        frames_embedding=torch.cat((frame_global,frame_local),2)
        frames_embedding=self.fc2(frames_embedding)
        vide_embed=self.en2(frames_embedding) 
        finish_time=time.time()
        print(finish_time-initial_time)
        return vide_embed
        