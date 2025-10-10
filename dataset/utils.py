from typing import List, Dict
import torch

def collate_fn(batch: List[Dict]):
    """
    自定义的 collate 函数，用于将 Dataset 返回的样本列表整理成批次。
    
    Args:
        batch: 一个列表，其中每个元素都是 Dataset.__getitem__ 返回的字典。
               例如: [{'frame': t1, 'audio': t2, 'label': l1}, {'frame': t3, ...}]
    """
    # 1. 将不同键的值分别收集到列表中
    frames = [item['frame'] for item in batch]
    audios = [item['audio'] for item in batch]
    labels = [item['label'] for item in batch]

    # 2. 对每个列表进行批处理（stacking）
    #    这里的逻辑和你原来的 strict_stack 类似，但更健壮
    
    # 检查是否是占位符，如果是，就不进行 stack
    if all(isinstance(f, torch.Tensor) and f.numel() > 1 for f in frames):
        batch_frame = torch.stack(frames, dim=0)
    else:
        # 如果是占位符或者混合了真实数据和占位符，你可能需要决定如何处理
        # 这里我们简单地返回列表，或者你可以创建一个批量的零张量
        batch_frame = frames # 或者根据需要进行其他处理

    if all(isinstance(a, torch.Tensor) and a.numel() > 1 for a in audios):
        batch_audio = torch.stack(audios, dim=0)
    else:
        batch_audio = audios
    
    # 标签通常总是存在的
    batch_label = torch.tensor(labels, dtype=torch.long)

    # 3. 返回一个包含批处理后数据的字典
    return {'frame': batch_frame, 'audio': batch_audio, 'label': batch_label}