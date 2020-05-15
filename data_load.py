# importing neceesery libraries

import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import os
import cv2
from skimage import io
from torchvision import transforms
import torch.nn.functional as F
import skimage







class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:  #(100,80).. 100 > 80 so newh , new_w = 100 * (100/80),240
                new_h, new_w = self.output_size * h / w, self.output_size   
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}






class FaceData(Dataset):
  def __init__(self,point_file,image_folder,transform=None):
      self.df = pd.read_csv(point_file)
      self.image_folder = image_folder
      self.images = self.df['Unnamed: 0'].values
      self.transform = transform

  def __len__(self):
      return len(self.images)

  def __getitem__(self,idx):

      img = self.images[idx]
      image = io.imread(os.path.join(self.image_folder,img))
     
      # image = mpimg.imread(os.path.join(self.image_folder,img))
      # if image has an alpha color channel, get rid of it
      if(image.shape[2] == 4):
            image = image[:,:,0:3]

      image = np.float32(image)

      key_pts = self.df[self.df['Unnamed: 0'] == img].values[0][1:].astype('float').reshape(-1,2)

      # key_pts = self.df.iloc[idx, 1:].as_matrix()
      # key_pts = key_pts.astype('float').reshape(-1, 2)


      sample = {'image': image, 'keypoints': key_pts}

      if self.transform:
          sample = self.transform(sample)

      return sample