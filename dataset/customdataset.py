import pandas as pd
import numpy as np 
import os
import torch
import cv2
from tqdm.notebook import tqdm
from glob import glob
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dataset

label_classes = {'Battery':0, 'Clothing':1, 'Glass':2, 'Metal':3, 'Paper':4, 'Paperpack':5, 'Plastic':6, 'Plasticbag':7, 'Styrofoam':8}

class CustomDataset(Dataset):
    def __init__(self, data_dir, train_mode, transforms):
        super().__init__()
        self.img_list = glob(os.path.join(data_dir, '*/*.*'))
        self.img_label = []
        self.transforms = transforms
        self.train_mode = train_mode
        if self.train_mode:
          for img_path in tqdm(self.img_list):
            label_name = img_path.split('/')[-2]
            self.img_label.append(label_classes[label_name])
        
    def __getitem__(self, index: int):
        img_path=self.img_list[index]
        file_name=img_path.split('/')[-1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = self.transforms(image=image)
        if self.train_mode:
          return image['image'], self.img_label[index]
        else:
          return image['image'], file_name
    
    def __len__(self) -> int:
        return len(self.img_list)