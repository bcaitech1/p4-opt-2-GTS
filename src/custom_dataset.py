import os
from glob import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
# Modify Your Custom Dataset
# TODO
label_classes = {'Battery':0, 'Clothing':1, 'Glass':2, 'Metal':3, 'Paper':4, 'Paperpack':5, 'Plastic':6, 'Plasticbag':7, 'Styrofoam':8}
class CustomDataset(Dataset):
    def __init__(self, data_dir, transforms):
        super().__init__()
        self.img_list = glob(os.path.join(data_dir, '*/*.*'))
        self.transforms = transforms
        
    def __getitem__(self, index: int):
        img_path=self.img_list[index]
        file_name=img_path.split('/')[-1]
        label_name=img_path.split('/')[-2]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = self.transforms(image=image)
        return image['image'], label_classes[label_name]
    
    def __len__(self) -> int:
        return len(self.img_list)