# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:48:25 2019

@author: LuoHan
"""
import numpy as np
import os
import torch
from albumentations import Compose, RandomCrop
from torch.utils.data import Dataset, DataLoader
from imageio import imread

class MyDataSet(Dataset):
    """
    """
    def __init__(self, root, patch_size = 384):
        super(MyDataSet, self).__init__()
        
        self.patch_size = patch_size
        self.padding_size = patch_size // 2
        
        self.root = root
        self.train_img_path = os.path.join(self.root, 'train')
        self.train_label_path = os.path.join(self.root, 'labels/train')
        self.GetImgid()
        
        self.files = []
        for i_id in self.img_ids:
            img_file = os.path.join(self.train_img_path,"%s.png" % i_id)
            label_file = os.path.join(self.train_label_path, "%s.tiff" % i_id)
            self.files.append({
                    'img' : img_file,
                    'label' : label_file,
                    'id' : i_id
                    })
        self.crop = self.__crop()
    
    def GetImgid(self):
        """
        """
        files = os.listdir(self.train_img_path)
        self.img_ids = [i_id.rstrip('.png') for i_id in files]
    
    def __len__(self):
        return len(self.files)
    
    def __crop(self):
        return Compose([RandomCrop(self.patch_size, self.patch_size)], p=1.0)
    
    def __padding(self, image):
        pad = ((self.padding_size, self.padding_size),
               (self.padding_size, self.padding_size))
        image = np.pad(image, pad, 'constant')
        return image
    
    def __getitem__(self, index):
        datafile = self.files[index]
        
        #load data
        #i_id = datafile['id']
        image = imread(datafile['img'])
        #灰度图只用一个通道
        image = image[...,0]
        label = imread(datafile['label'])
        label = 1 - label / 255.0
        label[label >= 0.5] = 1
        label[label < 0.5] = 0
        image = (image - image.mean()) / image.std(ddof=1)
        
        image = self.__padding(image)
        label = self.__padding(label)
        
        images, labels = [], []
        for i in range(20):
            crop = self.crop(image = image, mask = label)
            image_crop, label_crop = crop['image'], crop['mask']
            image_crop = np.expand_dims(image_crop, 0)
            images.append(image_crop)
            labels.append(label_crop)
        
        return torch.FloatTensor(images), torch.LongTensor(labels)


def MyDataLoader(root, batch_size):
    """
    """
    datasets = MyDataSet(root)
    dataloader = DataLoader(datasets, batch_size=batch_size,
                            shuffle=True)
    
    return dataloader

