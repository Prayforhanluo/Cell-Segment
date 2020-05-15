# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:55:29 2020

@author: LuoHan
"""

import os
from utils import *
from train import Trainer
from predict import Predictor
from loader import MyDataLoader


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    test_dir = './val'
    train_dir = './'

    models_dir = './models'
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    trainset = MyDataLoader('./', batch_size = 1)
    trainer = Trainer()
    model_path = trainer.run(trainset, models_dir)

    testset = []
    test_files = os.listdir(test_dir)
    for test_file in test_files:
        image_path = os.path.join(test_dir, test_file)
        testset.append(image_path)


    predictor = Predictor(model_path=model_path)
    output_dir = './valid'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    predictor.run(testset, output_dir)
    

    