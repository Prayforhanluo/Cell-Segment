# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:35:57 2020

@author: LuoHan
"""

import os
import torch
import numpy as np
from model import UNet
from loader import MyDataLoader
from imageio import imread, imwrite
from itertools import product
from tqdm import tqdm


class Predictor(object):
    """
    """
    MEAN, STD = 0.52, 0.19
    def __init__(self, model_path):
        self.model = UNet(1, 1, 16)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        self.model.eval()

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()
        
        return
    
    def __padding(self, image, padding_size):
        pad = ((padding_size, padding_size),
               (padding_size, padding_size))
        image = np.pad(image, pad, 'constant')
        return image

    def run(self, testset, output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for image_path in tqdm(testset, ncols=75):
            image_src = imread(image_path).astype(float)[..., 0] / 255.0
            image = (image_src - self.MEAN) / self.STD
            pad, gap = 384 // 2, 128
            image = self.__padding(image, pad)
            mask, count = np.zeros(image.shape), np.zeros(image.shape)

            centers = [0] + [gap * i - 1 for i in range(1, int(1024 / gap))] + [1023]
            centers = list(product(centers, centers))

            for c in tqdm(centers, ncols=75):
                image_crop = image[c[0]:c[0] + 384, c[1]:c[1] + 384]
                image_crop = np.expand_dims(image_crop, 0)
                image_crop = np.expand_dims(image_crop, 0)
                image_tensor = torch.Tensor(image_crop)
                if self.cuda:
                    image_tensor = image_tensor.cuda()
                image_pred = self.model(image_tensor)
                image_pred = torch.sigmoid(image_pred)
                image_pred = image_pred.squeeze(0)
                image_pred = image_pred.data.cpu().numpy()
                image_pred[image_pred > 0.5] = 1
                image_pred[image_pred <= 0.5] = 0

                mask[c[0]:c[0] + 384, c[1]:c[1] + 384] = image_pred
                count[c[0]:c[0] + 384, c[1]:c[1] + 384] += np.ones((384, 384))

            count[count == 0] += 1e-6
            mask /= count
            mask[mask > 0] = 1
            mask = mask[384 // 2 - 1:384 // 2 + 1023, 384 // 2 - 1:384 // 2 + 1023]
            mask = (255 - mask * 255).astype(np.uint8)

            mask_file = image_path.split('/')[-1].split('.')[0] + '.tiff'
            mask_path = os.path.join(output_dir, mask_file)
            imwrite(mask_path, mask)

        return
    
