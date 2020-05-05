import pandas as pd 
import os
import sys
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
#from src import audio_dataset
from PIL import Image
import torch

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
        
    
class Video_Frame_Data(Dataset):
    def __init__(self, csv_file,base_path_v=None,base_path_a=None,strict_num=16,strict_name=True,name_format=9,transform=None):
        self._table = pd.read_csv(csv_file,delimiter=' ')
        self.strict_num = strict_num
        self._base_path_v=base_path_v
        self._base_path_a=base_path_a
        if strict_name:
            self.name_format=name_format


        self.transform=transforms.Compose([
                     transforms.Resize((256,256)),
                     #transforms.Grayscale(),
                     transforms.ToTensor(),   
                     #transforms.Normalize((0.5, ), (0.5, ))])
                     transforms.Normalize((0.5,0.5,0.5 ), (0.5, 0.5,0.5))])
       # print(self._table.set_index(["Attribute","Wav_Path"]).count(level="Attribute"))
    def __len__(self):
        return len(self._table)

    def __getitem__(self, idx):
        folder_name = os.path.join(self._base_path_v,self._table.Vid_name[idx])
        first=True
        
        #audio_img=audio_dataset.plotstft(os.path.join(self._base_path_a,self._table.Attribute[idx],
        #                                    str(self._table.file_path[idx]).zfill(self.name_format)+'.wav'))
        audio_img=np.zeros((1,1))
        
        
        ###Read Frames
        START_PAD = Image.new(mode = "RGB", size = (256, 256), color = (255, 255, 255) )
        END_PAD= Image.new(mode = "RGB", size = (256, 256), color =(0, 0, 0))
        
        img=self.transform(START_PAD).unsqueeze(0)
        END_img=self.transform(END_PAD).unsqueeze(0)
        frame_len=len(os.listdir(folder_name))
        
        if frame_len<self.strict_num-2:
            for index_0 in range(frame_len):
                tempimg=Image.open(os.path.join(folder_name,os.listdir(folder_name)[index_0]))#.convert('L')       
                tempimg=self.transform(tempimg)
                tempimg=tempimg.unsqueeze(0)
                img=torch.cat((img,tempimg))
            for index_1 in range(self.strict_num-2-frame_len):
                img=torch.cat((img,END_img))
                
        else:    
            frame_index=(np.linspace(0,frame_len-1,self.strict_num-2,dtype=int))

            for index_2 in frame_index:
                tempimg=Image.open(os.path.join(folder_name,os.listdir(folder_name)[index_2]))#.convert('L')
                tempimg=self.transform(tempimg)
                tempimg=tempimg.unsqueeze(0)
                img=torch.cat((img,tempimg))

        labels = torch.from_numpy(np.array(self._table.Label[idx]))-1
            
     
            
        img=torch.cat((img,END_img))
        img=img[1: -1]
        return (img,audio_img,labels)
    