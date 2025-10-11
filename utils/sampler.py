# in utils/sampler.py

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from typing import List, Dict

class ClassSampler:
    """
    一个用于从数据集中按类别高效采样样本【索引】的工具类。
    
    这个类的逻辑完全基于您提供的 get_aud_images 函数。
    """
    def __init__(self, dataset: Dataset):
        """
        初始化采样器。
        它会遍历一次数据集，构建一个从类别到其所有样本索引的映射。

        Args:
            dataset (Dataset): 需要从中采样的 PyTorch 数据集。
        """
        self.dataset = dataset
        self.num_classes = getattr(dataset, 'num_classes', None)
        
        print("Initializing ClassSampler by mapping all sample indices to their classes...")
        self.indices_per_class: Dict[int, List[int]] = self._get_indices_per_class()
        print("ClassSampler is ready.")

    def _get_indices_per_class(self) -> Dict[int, List[int]]:
        """
        遍历数据集，构建 {类别: [索引1, 索引2, ...]} 的映射。
        这对应于旧代码中 `indices_class` 的构建过程。
        """
        # 如果 dataset 没有 num_classes 属性，我们动态计算一下
        if self.num_classes is None:
            print("Dataset has no 'num_classes' attribute, dynamically calculating...")
            all_labels = [self.dataset[i]['label'] for i in range(len(self.dataset))]
            self.num_classes = max(all_labels) + 1
            
        indices_map = {c: [] for c in range(self.num_classes)}

        for i in tqdm(range(len(self.dataset)), desc="Mapping indices to classes"):
            label = self.dataset[i]['label']
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            if label in indices_map:
                indices_map[label].append(i)
        
        return indices_map

    def sample(self, class_idx: int, num_samples: int) -> List[int]:
        """
        为一个给定的类别随机采样指定数量的样本【索引】。
        
        这完全复现了 get_aud_images 函数中的 np.random.permutation 逻辑。

        Args:
            class_idx (int): 类的索引。
            num_samples (int): 需要采样的样本数量。

        Returns:
            List[int]: 采样出的样本索引的 Python 列表。
        """
        # 获取该类别的所有可用索引
        available_indices = self.indices_per_class.get(class_idx, [])
        
        # 如果该类别没有任何样本，返回空列表
        if not available_indices:
            print(f"Warning: No samples found for class {class_idx}.")
            return []
            
        # --- 核心采样逻辑，与旧代码 np.random.permutation(...).tolist() 完全一致 ---
        
        # 如果请求的样本数少于或等于可用样本数，进行无重复采样
        if num_samples <= len(available_indices):
            # np.random.permutation(list)[:n] 是高效的无重复随机采样
            return np.random.permutation(available_indices)[:num_samples].tolist()
        else:
            # 如果请求的样本数多于可用样本数，旧代码的行为是返回所有可用样本。
            # 另一个选择是进行有重复的采样，我们先忠于原作。
            print(f"Warning: Requested {num_samples} for class {class_idx}, but only {len(available_indices)} are available. Returning all available samples shuffled.")
            return np.random.permutation(available_indices).tolist()

    # (可选) 一个更灵活的 sample 方法，允许重复采样
    def sample_with_replacement(self, class_idx: int, num_samples: int) -> List[int]:
        """允许在样本不足时进行重复采样，以满足 num_samples 的要求。"""
        available_indices = self.indices_per_class.get(class_idx, [])
        if not available_indices:
            return []
        # np.random.choice 提供了 replace 参数
        return np.random.choice(available_indices, size=num_samples, replace=True).tolist()