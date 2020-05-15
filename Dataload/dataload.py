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
class pre_train_data(Dataset):
    def __init__(self, csv_file,path):
        self._table=pd.read_csv(csv_file)
        self.path=path
        self.transform=transforms.Compose([
                     transforms.Resize((128,128)),
                     transforms.ToTensor(),                
                     transforms.Normalize((0.5,), (0.5,))])
    def __len__(self):
        return len(self._table)

    def __getitem__(self, idx):
        folder_name = self.path
        img=Image.open(os.path.join(folder_name,self._table.file_path[idx])).convert('L')

        img=self.transform(img)
        
        return img
        
class Video_Embedding_Data(Dataset):
    def __init__(self, embed_file,label_file,frame=24):
        self._table_embedding = pd.read_csv(embed_file)
        self._table_label = pd.read_csv(label_file,delimiter=' ')
        self.frame=frame
        
    def __len__(self):
        return len(self._table_embedding)

    def __getitem__(self, idx):
        embedding=torch.from_numpy(np.array(self._table_embedding.Embedding[idx].split(),dtype=float)).reshape((self.frame,-1))
        
        labels = torch.from_numpy(np.array(self._table_label.Label[idx]))-1
        empty=0
        return (embedding,empty,labels.long())
    
    
class Video_Frame_Data(Dataset):
    def __init__(self,csv_file,sub_csv_file=None,
                 base_path_v=None,base_path_a=None,frame_num=16,strict_name=True,name_format=9,embedding=False):
        
        self.max_frame_num=24
        self._table = pd.read_csv(csv_file,delimiter=' ')
        if sub_csv_file is None:
            self._table_embedding=None
        else:
            self._table_embedding=pd.read_csv(sub_csv_file)
        self.frame_num = frame_num
        self._base_path_v=base_path_v
        self._base_path_a=base_path_a
        self.embedding=embedding
        if strict_name:
            self.name_format=name_format
            

        self.transform=transforms.Compose([
                     transforms.Resize((256,256)),
                     transforms.ToTensor(),   
                     transforms.Normalize((0.5,0.5,0.5 ), (0.5, 0.5,0.5))])
        self.endPad=self.transform(Image.new(mode='RGB', size=(256,256), color=0))
    def __len__(self):
        return len(self._table)

    def __getitem__(self, idx):
        folder_name = os.path.join(self._base_path_v,self._table.Vid_name[idx])
        first=True
        audio_img=np.zeros((1,1))
        labels = torch.from_numpy(np.array(self._table.Label[idx]))-1
            
        if self._table_embedding is not None:
            temp_frame_embedding=torch.from_numpy(np.array(self._table_embedding.Embedding[idx].split(),dtype=float)).reshape((self.max_frame_num,-1))
            frame_data=torch.empty(size=(self.frame_num,1000),dtype=torch.double)
            if self.frame_num<25:
                index=np.linspace(0,23,self.frame_num,dtype=int)
 
                for i,copy in enumerate(index):
                    frame_data[i]=temp_frame_embedding[copy]
            return (frame_data,audio_img,labels)
    

            
        frame_raw_list=os.listdir(folder_name)
        frame_len=len(frame_raw_list)

        frame_raw_list=sorted(frame_raw_list)

        frame_list=[]
        if frame_len<self.frame_num:
            for index_0 in range(frame_len):
                frame_path=os.path.join(folder_name,frame_raw_list[index_0])               
                tempimg=Image.open(frame_path)       
                frame_list.append(self.transform(tempimg))
            
        else:    
            frame_index=(np.linspace(0,frame_len-1,self.frame_num,dtype=int))

            for index_2 in frame_index:
                frame_path=os.path.join(folder_name,frame_raw_list[index_2])
                tempimg=Image.open(frame_path)
                frame_list.append(self.transform(tempimg))
        while(len(frame_list)<self.frame_num):
            frame_list.append(self.endPad)
        frame_data=torch.stack(frame_list,dim=0)
            
        return (frame_data,audio_img,labels)
    
class multi_frames():
    def __init__(self,frame_list,channel=3,img_width=256,img_length=256,embed_dim=1000,frame_embedding=None):
        self.frame_list=frame_list
        
        if frame_embedding is None: 
            self.frames = torch.empty(size=(len(frame_list), channel, img_width,img_length)) 
            for index,element in enumerate(frame_list):
                self.frames[index]=element.frame_img
        else:
            self.frames=frame_embedding
    def __len__(self):
        return len(self.frame_list)

        

class frame():
    def __init__(self,frame_img,frame_name,max_face=5):
        self.frame_img=frame_img
        self._file_name=frame_name
        self.transform_face=transforms.Compose([
                                 transforms.Resize((32,32)), 
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5 ), (0.5, 0.5,0.5))])
        
        self.transform_frame=transforms.Compose([
                     transforms.Resize((256,256)),
                     transforms.ToTensor(),   
                     transforms.Normalize((0.5,0.5,0.5 ), (0.5, 0.5,0.5))])

        
        def face_extraction(img,max_face):
            bounding_boxes, landmarks = detect_faces(img)
            img_list=[]
            for box_index,(left,right,up,bottom,_) in enumerate(bounding_boxes):
                cropped_img=img.crop((left,right,up,bottom))
                img_list.append(self.transform_face(cropped_img))
                if len(img_list)==max_face:
                    break
            while len(img_list) !=max_face:
                END_PAD= Image.new(mode = "RGB", size = (256, 256), color =(0, 0, 0))
                img_list.append(self.transform_face(END_PAD))
            img_list=torch.stack(img_list)

            return img_list
        self.faces=face_extraction(self.frame_img,max_face)


        
        self.frame_img=self.transform_frame(frame_img)

        
        
    
        
class Frame_Face(Dataset):
    def __init__(self,csv_file,sub_csv_file=None,
                 base_path_v=None,base_path_a=None,frame_num=16,strict_name=True,name_format=9,embedding=False):
        
        self.max_frame_num=24
        self._table = pd.read_csv(csv_file,delimiter=' ')
        if sub_csv_file is None:
            self._table_embedding=None
        else:
            self._table_embedding=pd.read_csv(sub_csv_file)
        self.frame_num = frame_num
        self._base_path_v=base_path_v
        self._base_path_a=base_path_a
        self.embedding=embedding
        if strict_name:
            self.name_format=name_format
            

        self.transform=transforms.Compose([
                     transforms.Resize((256,256)),
                     transforms.ToTensor(),   
                     transforms.Normalize((0.5,0.5,0.5 ), (0.5, 0.5,0.5))])
        self.endPad=self.transform(Image.new(mode='RGB', size=(256,256), color=0))