import os
import scipy.io
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Images(Dataset):
    def __init__(self, label_file, root_dir, transform=None):
        self.labels = scipy.io.loadmat(os.path.join(root_dir,label_file))['params']
        self.root_dir = root_dir
        self.transform=transform
        
    def __len__(self):
        return self.labels.shape[1]  # 1000
    
    def __getitem__(self,index):
        filename = 'image'+str(index)+'.mat'
        img_path = os.path.join(self.root_dir,filename)
        mat = scipy.io.loadmat(img_path)
        image = mat['data']
        y_label = torch.tensor(self.labels[:,index])
        
        if self.transform:
            image = self.transform(image)
            
        return (image, y_label)
         
# data = Images(label_file='train_labels.mat', root_dir='/data/tangm6/images', transform=transforms.ToTensor())
# print("Dataset length: {}".format(len(data)))
