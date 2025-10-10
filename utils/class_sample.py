# 你可以将这个类放在一个新的文件里，比如 utils/sampler.py

import torch
import numpy as np
import pickle
from tqdm import tqdm
import torch.nn.functional as F

class ClassSampler:
    """
    一个用于从数据集中按类别采样样本的工具类。
    """
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset
        print("Initializing ClassSampler for random sampling...")
        self.indices_per_class = self._get_indices_per_class()

    def _get_indices_per_class(self) -> dict:
        """
        遍历数据集一次，构建一个从类别到索引列表的映射。
        """
        indices_map = {}
        # 假设数据集有 num_classes 属性
        num_classes = getattr(self.dataset, 'num_classes', None)
        if num_classes is None:
            # 如果没有，就动态计算
            all_labels = [self.dataset[i]['label'] for i in range(len(self.dataset))]
            num_classes = max(all_labels) + 1
            
        for c in range(num_classes):
            indices_map[c] = []

        # 使用 tqdm 显示进度
        for i in tqdm(range(len(self.dataset)), desc="Mapping indices to classes"):
            label = self.dataset[i]['label']
            # 支持 label 是 tensor 的情况
            if isinstance(label, torch.Tensor):
                label = label.item()
            indices_map[label].append(i)
        
        return indices_map

    def sample(self, class_idx: int, num_samples: int) -> list:
        # 只保留随机采样逻辑
        available_indices = self.indices_per_class.get(class_idx, [])
        if not available_indices:
            return []
            
        if len(available_indices) < num_samples:
            # 如果样本不够，可以重复采样 (with replacement)
            return np.random.choice(available_indices, num_samples, replace=True).tolist()
        else:
            # 如果样本足够，就不重复采样
            return np.random.choice(available_indices, num_samples, replace=False).tolist()

def downscale(image_syn, scale_factor):
    image_syn = F.upsample(image_syn, scale_factor=scale_factor, mode='bilinear')
    return image_syn

