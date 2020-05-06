from torch.utils.data import Dataset, DataLoader
from src import detect_faces, show_bboxes
import torch
import torch.nn as nn
from torchvision import transforms


class Facial(Dataset):
    def __init__(self,image,max_number):
        self.img=img
        self.max_number=max_number
        self.transform1=transforms.Compose([
                                 transforms.Resize((64,64)), 
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5 ), (0.5, 0.5,0.5))])
        self.face=self.face_extraction(img)

    def face_extraction(self,img):
        bounding_boxes, landmarks = detect_faces(img)
        img_list=[]
        for box_index,(left,right,up,bottom,_) in enumerate(bounding_boxes):
            cropped_img=img.crop((left,right,up,bottom))
            img_list.append(self.transform1(cropped_img))
            if len(img_list)==max_face:
                break
        while len(img_list) !=max_face:
            END_PAD= Image.new(mode = "RGB", size = (256, 256), color =(0, 0, 0))
            img_list.append(self.transform1(END_PAD))
        img_list=torch.stack(img_list)
        
        return img_list