# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:51:31 2019

@author: LuoHan
"""

import os
import torch
import numpy as np
import torch.optim as optim
from radam import RAdam
from model import UNet
from utils import DiceLoss, CombineLoss, FocalLoss2d
from loader import MyDataLoader
from sklearn.metrics import f1_score


class Trainer(object):
    """
    """
    def __init__(self):
        torch.set_num_threads(4)
        self.n_epochs = 10
        self.batch_size = 1
        self.patch_size = 384
        self.is_augment = False
        self.cuda = torch.cuda.is_available()
        self.__build_model()
        
    
    def __build_model(self):
        self.model = UNet(1,1, base=16)
        if self.cuda:
            self.model = self.model.cuda()
    
    def __reshapetensor(self, tensor, itype = 'image'):
        if itype == 'image':
            d0, d1, d2, d3, d4 = tensor.size()
            tensor = tensor.view(d0 * d1, d2, d3, d4)
        else:
            d0, d1, d2, d3 = tensor.size()
            tensor = tensor.view(d0 * d1, d2, d3)
        
        return tensor
    
    def __get_optimizer(self, **params):
        opt_params = {
            'params': self.model.parameters(),
            'lr': 1e-2, 'weight_decay': 1e-5
        }
        self.optimizer = RAdam(**opt_params)

        # self.scheduler = None
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', factor=0.5,
            patience=10, verbose=True, min_lr=1e-5
        )
    
    def run(self, trainset, model_dir):
        """
        """
        print('='*100)
        print('Trainning model')
        print('='*100)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        
        model_path = os.path.join(model_dir, 'model.pth')
        
        #loss_fn = DiceLoss()
        loss_fn = FocalLoss2d()
        #loss_fn = CombineLoss({'dice':0.5, 'focal':0.5})

        self.__get_optimizer()
        Loss = []
        F1 = []
        for epoch in range(self.n_epochs):
           
            for ith_batch, data in enumerate(trainset):
                images, labels = [d.cuda() for d in data] if self.cuda else data
                images = self.__reshapetensor(images, itype='image')
                labels = self.__reshapetensor(labels, itype='label')
                
                preds = self.model(images)
                loss = loss_fn(preds, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                Loss.append(loss.item())
                preds = torch.sigmoid(preds)
                preds[preds > 0.5] = 1
                preds[preds <= 0.5] = 0
                preds = preds.cpu().detach().numpy().flatten()
                labels = labels.cpu().detach().numpy().flatten()
                f1 = f1_score(labels, preds, average='binary')
                F1.append(f1)
                
                print('EPOCH : {}-----BATCH : {}-----LOSS : {}-----F1 : {}'.format(
                    epoch, ith_batch, loss.item(), f1))
        
        torch.save(self.model.state_dict(), model_path)
        
        return model_path
           
                




    