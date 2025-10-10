import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
from registry import DATASET  # 假设 DATASETS 是一个 Registry 实例
from base import BaseDataset # 假设你有一个 BaseDataset 抽象基类
from utils import read_cfg   # 假设你有一个读取 yaml 的函数

# ==============================================================================
# 1. 定义标准的 PyTorch Dataset 类
#    职责：根据索引提供单个数据样本。
# ==============================================================================

class AVEDataset(Dataset):
    """
    一个标准的 PyTorch Dataset，用于处理 AVE 数据集。

    这个类的职责是根据索引高效地提供单个数据样本。
    它假设所有的数据（音频、视频、标签）已经被加载到内存中。
    """
    def __init__(self, 
                 labels: torch.Tensor,
                 audio_data: Optional[torch.Tensor] = None, 
                 image_data: Optional[torch.Tensor] = None):
        """
        初始化数据集。

        Args:
            labels (torch.Tensor): 标签张量。
            audio_data (Optional[torch.Tensor]): 音频数据张量。如果模态不包含音频，则为 None。
            image_data (Optional[torch.Tensor]): 视频帧数据张量。如果模态不包含视频，则为 None。
        """
        super().__init__()
        
        # 存储传入的数据
        self.labels = labels
        self.audio = audio_data
        self.images = image_data

        # 安全性检查：确保至少有一种模态的数据存在
        if self.audio is None and self.images is None:
            raise ValueError("AVEDataset initialized without any data. At least one of 'audio_data' or 'image_data' must be provided.")

    def __len__(self) -> int:
        """返回数据集中样本的总数。"""
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        根据给定的索引，获取一个数据样本。这是 DataLoader 在后台调用的核心方法。

        Args:
            index (int): 样本的索引。

        Returns:
            Dict[str, Any]: 包含单个样本数据的字典。
        """
        label = self.labels[index]
        
        # 如果某个模态的数据不存在 (为 None)，则返回一个占位符张量。
        # 这个占位符将在 collate_fn 中被处理。
        audio_sample = self.audio[index] if self.audio is not None else torch.tensor(0.0)
        frame_sample = self.images[index] if self.images is not None else torch.tensor(0.0)
        
        return {
            'frame': frame_sample, 
            'audio': audio_sample, 
            'label': label
        }

# ==============================================================================
# 2. 定义数据集的构建器 (Builder/Factory) 类
#    职责：读取配置，加载数据，并创建上面的 Dataset 实例。
# ==============================================================================

@DATASET.register('AVE')
class AVEBuilder(BaseDataset):
    """
    AVE 数据集的构建器 (Builder)。

    这个类被注册到全局的 DATASETS 注册表中，并由工厂函数调用。
    它的 `build` 方法负责创建并返回一个功能完整的 AVEDataset 实例。
    """
    def __init__(self, config_path: str = "config/dataset/AVE.yaml"):
        """
        初始化构建器，主要是加载和解析配置文件。
        这个阶段不执行任何耗时的数据加载操作。

        Args:
            config_path (str): 数据集配置文件的路径。
        """
        super().__init__()
        
        self.cfg = read_cfg(path=config_path)
        
        # 从配置中解析出关键参数
        self.input_modality = self.cfg.get('input_modality', 'av')
        self.im_channel = self.cfg.get('im_channel', 3)
        self.mean = self.cfg.get('mean', [0.485, 0.456, 0.406])
        self.std = self.cfg.get('std', [0.229, 0.224, 0.225])

        # 将配置存储起来，以便 build 方法使用
        self.train_data_path = self.cfg.get('train_data_path')
        self.test_data_path = self.cfg.get('test_data_path')
        self.aud_tag = self.cfg.get('aud_tag')
        self.im_tag = self.cfg.get('im_tag')
        self.label_tag = self.cfg.get('label_tag')

    def build(self, mode: str = 'train') -> AVEDataset:
        """
        构建并返回一个 AVEDataset 实例。
        这个方法会执行实际的数据加载（I/O 操作）。

        Args:
            mode (str): 'train' 或 'test'，决定加载哪个数据集。

        Returns:
            AVEDataset: 一个配置好的、包含数据的 AVEDataset 实例。
        """
        assert mode in ['train', 'test'], f"Invalid mode '{mode}'. Must be 'train' or 'test'."
        
        # 1. 根据 mode 选择正确的数据路径
        data_path = self.train_data_path if mode == 'train' else self.test_data_path
        if not data_path:
            raise ValueError(f"'{mode}_data_path' not specified in the config file.")
            
        print(f"Building '{mode}' dataset from: {data_path}")
        
        # 2. 从磁盘加载原始数据文件
        raw_data = torch.load(data_path, map_location='cpu')
        
        # 3. 提取各个部分
        audio_data = raw_data.get(self.aud_tag)
        image_data = raw_data.get(self.im_tag)
        labels_data = raw_data.get(self.label_tag)

        if labels_data is None:
            raise KeyError(f"Label tag '{self.label_tag}' not found in the data file: {data_path}")
        
        # 4. 根据 input_modality 决定要传递给 AVEDataset 的数据
        final_audio_data = None
        if self.input_modality in ('a', 'av'):
            if audio_data is None:
                raise KeyError(f"Audio tag '{self.aud_tag}' not found, but modality is '{self.input_modality}'.")
            final_audio_data = audio_data.detach().float()

        final_image_data = None
        if self.input_modality in ('v', 'av'):
            if image_data is None:
                raise KeyError(f"Image tag '{self.im_tag}' not found, but modality is '{self.input_modality}'.")
            final_image_data = image_data.detach().float() / 255.0 # 归一化
            for c in range(self.im_channel):
                final_image_data[:, c] = (final_image_data[:, c] - self.mean[c]) / self.std[c]

        # 5. 创建并返回最终的 AVEDataset 实例
        dataset_instance = AVEDataset(
            labels=labels_data.detach(),
            audio_data=final_audio_data,
            image_data=final_image_data
        )

        # print(f"Successfully built '{mode}' dataset with {len(dataset_instance)} samples.")
        return dataset_instance