from registry import *

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