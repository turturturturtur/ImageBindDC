from registry import *
from typing import Optional

def create_model(name:str,**kwargs):
    '''
    通过名字创建模型
    '''
    model_class = MODEL.get(name)
    model = model_class(**kwargs)
    return model

def create_dataset(name:str,**kwargs):
    '''
    通过名字创建数据集
    '''
    dataset_class = DATASET.get(name)
    dataset = dataset_class(**kwargs)
    return dataset

def create_loss(name:str,**kwargs):
    '''
    通过名字创建损失函数
    '''
    loss_class = LOSS.get(name)
    loss = loss_class(**kwargs)
    return loss