import torch
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build():
        pass

    def __getitem__(self, index: int):
        """
        __getitem__ 的黄金法则:
        1. 接收一个整数索引。
        2. 返回一个样本的数据，通常是一个字典。
        3. 保持简单！
        """
        # 1. 根据索引获取单个数据项
        label = self.labels[index]
        
        # 2. 根据模态决定是加载数据还是返回占位符
        #    注意：返回一个有形状的零张量通常比返回一个标量 torch.zeros(1) 更好，
        #    除非你的 collate_fn 会特殊处理它。
        #    这里我们假设下游 collate_fn 知道如何处理。
        audio = self.audio[index] if self.input_modality in ('a', 'av') else torch.tensor(0.)
        frame = self.images[index] if self.input_modality in ('v', 'av') else torch.tensor(0.)
            
        # 3. 将单个样本打包成字典返回
        return {'frame': frame, 'audio': audio, 'label': label}
    
    def __len__(self):
        return self.labels.shape[0]

    @property
    def num_class(self):
        return self.num_class