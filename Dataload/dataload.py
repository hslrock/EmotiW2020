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
    def __init__(self,csv_file,sub_csv_file=None,
                 base_path_v=None,face_path=None,base_path_a=None,frame_num=16,strict_name=True,name_format=9,embedding=False,direct=False,audio_csv=None,train_mode=True):
        
        self.max_frame_num=25
        self._audio_table=pd.read_csv(audio_csv)
        self._table = pd.read_csv(csv_file,delimiter=' ')
        if sub_csv_file is None:
            self._table_embedding=None
        else:
            self._table_embedding=pd.read_csv(sub_csv_file)
        self.frame_num = frame_num
        self._base_path_v=base_path_v
        self._base_path_a=base_path_a
        self.face_path=face_path
        self.face_num=5
        self.train_mode=train_mode
        self.embedding=embedding
        if strict_name:
            self.name_format=name_format
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
        face_folder_name = os.path.join(self.face_path,video_name)
        faces_path=face_folder_name+'.pt'
        temp_face_data=torch.load(faces_path)
 
        if self.frame_num<self.max_frame_num:
                face_data=torch.empty(size=(self.frame_num,self.face_num,3,64,64),dtype=torch.double)
                index=np.linspace(0,self.max_frame_num-1,self.frame_num,dtype=int)
                for i,copy in enumerate(index):
                    face_data[i]=temp_face_data[copy]
        else:
            face_data=temp_face_data
        
        folder_name = os.path.join(self._base_path_v,self._table.Vid_name[idx])
        audio_feature=self._audio_table.loc[self._audio_table.file_name ==self._table.Vid_name[idx]+'.arff']
        audio_feature=torch.from_numpy(pd.to_numeric(audio_feature.values[0][3:]))
        if self.train_mode:
            labels = torch.from_numpy(np.array(self._table.Label[idx]))-1
            
        if self._table_embedding is not None:
            temp_frame_embedding=torch.from_numpy(np.array(self._table_embedding.Embedding[idx].split(),dtype=float)).reshape((self.max_frame_num,-1))
            frame_data=torch.empty(size=(self.frame_num,1000),dtype=torch.double)
            if self.frame_num<25:
                index=np.linspace(0,23,self.frame_num,dtype=int)
 
                for i,copy in enumerate(index):
                    frame_data[i]=temp_frame_embedding[copy]
            return (frame_data,face_data,audio_feature,labels)
        
        if self.direct:
            
            video_path=folder_name+'.pt'
            
            temp_frame_data=torch.load(video_path)
            
            if self.frame_num<self.max_frame_num:
                frame_data=torch.empty(size=(self.frame_num,3,256,256),dtype=torch.double)
                index=np.linspace(0,self.max_frame_num-1,self.frame_num,dtype=int)
                for i,copy in enumerate(index):
                    frame_data[i]=temp_frame_data[copy]
            else:
                frame_data=temp_frame_data
                
            return (frame_data,face_data,audio_feature,labels)
          #  frame_data=torch.load(os.path.join(folder_name))

    

        


        
        
    
        
class Face_Data(Dataset):
    def __init__(self,csv_file,face_path=None):
        self.frame_num=25
        self.max_frame_num=25
        self._table = pd.read_csv(csv_file,delimiter=' ')
        self.transform=transforms.Compose([
                     transforms.ToTensor(),   
                     transforms.Normalize((0.5,0.5,0.5 ), (0.5, 0.5,0.5))])
        self.face_path=face_path


            
    def __getitem__(self, idx):
        folder_name = os.path.join(self.face_path,self._table.Vid_name[idx])
        frame_raw_list=os.listdir(folder_name)
        frame_len=len(frame_raw_list)

        frame_raw_list=sorted(frame_raw_list)
           # print(frame_raw_list)
        frame_list=[]
  
        frame_index=(np.linspace(0,frame_len-1,self.frame_num,dtype=int))
        for index_2 in frame_index:
            frame_path=os.path.join(folder_name,frame_raw_list[index_2])
            tempimg=Image.open(frame_path)
            frame_list.append(self.transform(tempimg))
        while(len(frame_list)<self.frame_num):
            frame_list.append(self.endPad)
        frame_data=torch.stack(frame_list,dim=0)
            
        return (frame_data,self._table.Vid_name[idx])
        
