
import numpy as np
import torch
import torch.nn as nn
import os
import cv2
from skimage import io
import matplotlib.image as mpimg
from torchvision import transforms
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
        super(Net, self).__init__()
       
        #defining maxpool block
        self.maxpool = nn.MaxPool2d(2, 2)
               
        #defining dropout block
        self.dropout = nn.Dropout(p=0.2)
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        #defining second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        #defining third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        #defining linear output layer
        self.fc1 = nn.Linear(128*26*26, 136)


  def forward(self, x):
        
        #passing tensor x through first conv layer
        x = self.maxpool(F.relu(self.conv1(x)))
     
        #passing tensor x through second conv layer
        x = self.maxpool(F.relu(self.conv2(x)))
        
        #passing tensor x through third conv layer
        x = self.maxpool(F.relu(self.conv3(x)))
        
      
        #flattening x tensor
        x = x.view(x.size(0), -1)
        
        #applying dropout
        x = self.dropout(x)
     
        #passing x through linear layer
        x = self.fc1(x)
        
        #returning x
        return x
