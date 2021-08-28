import os
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from dataset3D import Images
from cnn_approach1 import ConvNet
from torch.utils.data import Dataset, DataLoader


batch_size=4
root_dir = '/data/tangm6/images/'
data = Images(label_file='train_labels.mat', root_dir=root_dir, transform=transforms.ToTensor())
trainSet, testSet = torch.utils.data.random_split(data,[400,100])
train_loader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testSet, batch_size=batch_size, shuffle=True)

# evaluate model
net = ConvNet()
net.load_state_dict(torch.load('/data/tangm6/output_files/cnn_net.pth'))
net = net.cuda()

index = 0
error = np.zeros((6,100))
with torch.no_grad():
    for data in test_loader:
        image, label = data[0].cuda(), data[1].cuda()
        outputs = net(image)
        outputs = outputs.cpu().numpy()
        error[:,index] = abs(label-outputs)
        index += 1
print('Average test error:')
print(np.mean(error,1))
