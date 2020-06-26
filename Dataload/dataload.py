import pandas as pd 
import os
import sys
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src import detect_faces, show_bboxes
from PIL import Image
import torch

np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        

class Video_Frame_Data(Dataset):
    def __init__(self,csv_file,audio_csv=None,base_path_v=None,face_path=None,base_path_a=None,frame_num=16,
                 direct=False,train_mode=True):

        self._audio_table=pd.read_csv(audio_csv)
        self._table = pd.read_csv(csv_file,delimiter=' ')

        self.frame_num = frame_num
        self._base_path_v=base_path_v
        self._base_path_a=base_path_a
        self.face_path=face_path
        self.face_num=5
        self.train_mode=train_mode
        self.direct=direct

        self.transform=transforms.Compose([
                     transforms.Resize((256,256)),
                     transforms.ToTensor(),   
                     transforms.Normalize((0.5,0.5,0.5 ), (0.5, 0.5,0.5))])
        self.endPad=self.transform(Image.new(mode='RGB', size=(256,256), color=0))
    def __len__(self):
        return len(self._table)

    def __getitem__(self, idx):
        video_name=self._table.Vid_name[idx]
        
        video_path = os.path.join(self._base_path_v,video_name)+'.pt'
        #######################Loading Audio Data ########################
        audio_feature=self._audio_table.loc[self._audio_table.file_name ==self._table.Vid_name[idx]+'.arff']
        audio_feature=torch.from_numpy(pd.to_numeric(audio_feature.values[0][2:]))
        
        ###################Loading Label Data##################
        labels=None
        if self.train_mode:
            labels = torch.from_numpy(np.array(self._table.Label[idx]))-1
            
        ######################Loading Face Data Loading###################################
        faces_path = os.path.join(self.face_path,video_name)+'.pt'
        temp_face_data=torch.load(faces_path)
        
        if self.frame_num<self.temp_face_data.size(0):
                face_data=torch.empty(size=(self.frame_num,self.face_num,3,64,64),dtype=torch.double)
                index=np.linspace(0,temp_face_data.size(0)-1,self.frame_num,dtype=int)
                for i,copy in enumerate(index):
                    face_data[i]=temp_face_data[copy]
        else:
            face_data=temp_face_data
            
       ######################Loading Frame Data Loading###################################     
        video_path = os.path.join(self._base_path_v,video_name)+'.pt'
        temp_frame_data=torch.load(video_path)
        
        if self.frame_num<temp_frame_data.size(0):
            frame_data=torch.empty(size=(self.frame_num,3,256,256),dtype=torch.double)
            index=np.linspace(0,temp_frame_data.size(0)-1,self.frame_num,dtype=int)
            for i,copy in enumerate(index):
                frame_data[i]=temp_frame_data[copy]
        else:
            frame_data=temp_frame_data
        if self.train_mode:
            return (video_name,frame_data,face_data,audio_feature,labels)
        else:
            return (video_name,frame_data,face_data,audio_feature)


class Video_Frame_Only_Data(Dataset):
    def __init__(self,csv_file,
                 base_path_v=None,frame_num=16
                ,audio_csv=None,train_mode=True):
        
        self.max_frame_num=25
        self._audio_table=pd.read_csv(audio_csv)
        self._table = pd.read_csv(csv_file,delimiter=' ')
        self.frame_num = frame_num
        self._base_path_v=base_path_v
        self.face_num=5
        self.train_mode=train_mode

    def __len__(self):
        return len(self._table)

    def __getitem__(self, idx):
        video_name=self._table.Vid_name[idx]
        folder_name = os.path.join(self._base_path_v,self._table.Vid_name[idx])
       
        audio_feature=self._audio_table.loc[self._audio_table.file_name ==self._table.Vid_name[idx]+'.arff']
        audio_feature=torch.from_numpy(pd.to_numeric(audio_feature.values[0][2:]))
            
        video_path=folder_name+'.pt'
        temp_frame_data=torch.load(video_path)
        if self.frame_num<self.max_frame_num:
            frame_data=torch.empty(size=(self.frame_num,3,256,256),dtype=torch.double)
            index=np.linspace(0,temp_frame_data.size(0)-1,self.frame_num,dtype=int)
            for i,copy in enumerate(index):
                frame_data[i]=temp_frame_data[copy]
        else:
            frame_data=temp_frame_data
        if self.train_mode:
            labels = torch.from_numpy(np.array(self._table.Label[idx]))-1
            return (video_name,frame_data,audio_feature,labels)
        else:
            return (video_name,frame_data,audio_feature)
        
        
    

class Audio_Data(Dataset):
    def __init__(self,csv_file,audio_csv=None,train_mode=True):
        
        self.max_frame_num=25
        self._audio_table=pd.read_csv(audio_csv)
        self._table = pd.read_csv(csv_file,delimiter=' ')
 
        self.train_mode=train_mode

    def __len__(self):
        return len(self._table)

    def __getitem__(self, idx):
        video_name=self._table.Vid_name[idx]
        audio_feature=self._audio_table.loc[self._audio_table.file_name ==self._table.Vid_name[idx]+'.arff']
        audio_feature=torch.from_numpy(pd.to_numeric(audio_feature.values[0][3:]))
        if self.train_mode:
            labels = torch.from_numpy(np.array(self._table.Label[idx]))-1
        else:
            labels=None
        return (audio_feature,labels)