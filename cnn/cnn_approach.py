import os
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from dataset3D import Images
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# load dataset
batch_size=4
root_dir = '/data/tangm6/images/'
data = Images(label_file='train_labels.mat', root_dir=root_dir, transform=transforms.ToTensor())
trainSet, testSet = torch.utils.data.random_split(data,[1200,300])
train_loader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testSet, batch_size=batch_size, shuffle=True)
 
# define network
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(300, 600, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(600, 150, 5)
        self.conv3 = nn.Conv2d(150, 16, 5)
        self.fc1 = nn.Linear(16 * 28 * 28, 480)
        self.fc2 = nn.Linear(480, 120)
        self.fc3 = nn.Linear(120, 64)
        self.fc4 = nn.Linear(64, 6)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x))) 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
 
net = ConvNet()
if torch.cuda.is_available():
    print("Running with GPU.")
    net = net.cuda()
 
# define hyperparameters
lr=0.0001
momentum=0.9
epochs=5
 
# train network
criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
 
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        image, params = data[0].cuda(), data[1].cuda()
       
        # zero the parameter gradients
        optimizer.zero_grad()
 
        # forward + backward + optimize
        outputs = net(image)
        loss = criterion(outputs.float(), params.float())
        loss.backward()
        optimizer.step()
 
        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
 
print('Finished Training')
 
# save model
path = '/data/tangm6/output_files/cnn2_net.pth'
torch.save(net.state_dict(), path)
 
 
# evaluate model
index = 0
error = np.zeros((6,100))
with torch.no_grad():
    for data in test_loader:
        image, label = data[0].cuda(), data[1].cuda()
        outputs = net(image)
        outputs = outputs.cpu()
        label = label.cpu()
        error[:,index] = np.mean(abs(label.data.numpy()-outputs.data.numpy()),0)
        index += 1
print('Average test error:')
print(np.mean(error,1))
