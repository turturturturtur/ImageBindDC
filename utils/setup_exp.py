from torchvision.transforms import v2
import torch
from tqdm import tqdm
import numpy as np
from .class_sample import ClassSampler, downscale
from torch.utils.data import Dataset


def get_img_transform():
    img_transform = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ])
    return img_transform

def get_syn_data(dst_train: Dataset, dst_syn_container: Dataset, ipc: int) -> Dataset:
    """
    通过采样、缩放和拼接真实数据，来填充并返回一个合成数据集对象。
    这个函数本身是完整的，依赖于 ClassSampler 和 downscale。

    Args:
        dst_train (Dataset): 包含所有真实数据的源数据集。
        dst_syn_container (Dataset): 一个空的或模板数据集对象，其内部数据将被替换。
        ipc (int): Images Per Class，每个类别最终生成的合成样本数量。

    Returns:
        Dataset: 一个填充了合成数据的、与 dst_train 结构相同的 Dataset 对象。
    """
    # 1. 初始化 ClassSampler (依赖)
    sampler = ClassSampler(dataset=dst_train)
    num_classes = dst_train.num_classes
    device = dst_train.audio.device if hasattr(dst_train, 'audio') and dst_train.audio is not None else dst_train.images.device

    # 2. 创建空列表来收集每个类别处理好的数据块
    list_syn_aud = []
    list_syn_img = []
    list_syn_labels = []

    print("Dynamically building synthetic data...")
    for c in tqdm(range(num_classes), desc="Processing Classes"):
        # 3.1 & 3.2: 采样索引
        num_samples_to_get = ipc * 4
        selected_indices = sampler.sample(class_idx=c, num_samples=num_samples_to_get)
        
        if not selected_indices:
            print(f"Warning: No samples found for class {c}. Skipping this class.")
            continue
            
        # 3.3: 批量获取真实数据
        batch_data = dst_train[selected_indices]
        auds_real = batch_data['audio'].to(device)
        imgs_real = batch_data['frame'].to(device)
        
        if auds_real.dim() == 4: auds_real = auds_real.unsqueeze(2)

        # 3.4: 构建“四合一”数据块
        # 音频部分
        if auds_real is not None and auds_real.numel() > 0:
            aud_part1 = downscale(auds_real[0*ipc:1*ipc].squeeze(2), 0.5).unsqueeze(2)
            aud_part2 = downscale(auds_real[1*ipc:2*ipc].squeeze(2), 0.5).unsqueeze(2)
            aud_part3 = downscale(auds_real[2*ipc:3*ipc].squeeze(2), 0.5).unsqueeze(2)
            aud_part4 = downscale(auds_real[3*ipc:4*ipc].squeeze(2), 0.5).unsqueeze(2)
            _, _, _, a_half_h, a_half_w = aud_part1.shape
            aud_canvas_shape = (ipc, aud_part1.shape[1], 1, a_half_h * 2, a_half_w * 2)
            aud_canvas = torch.zeros(aud_canvas_shape, device=device)
            aud_canvas[:, :, :, :a_half_h, :a_half_w] = aud_part1
            aud_canvas[:, :, :, a_half_h:, :a_half_w] = aud_part2
            aud_canvas[:, :, :, :a_half_h, a_half_w:] = aud_part3
            aud_canvas[:, :, :, a_half_h:, a_half_w:] = aud_part4
            list_syn_aud.append(aud_canvas)

        # 图像部分
        if imgs_real is not None and imgs_real.numel() > 0:
            img_part1 = downscale(imgs_real[0*ipc:1*ipc], 0.5)
            img_part2 = downscale(imgs_real[1*ipc:2*ipc], 0.5)
            img_part3 = downscale(imgs_real[2*ipc:3*ipc], 0.5)
            img_part4 = downscale(imgs_real[3*ipc:4*ipc], 0.5)
            _, _, v_half_h, v_half_w = img_part1.shape
            img_canvas_shape = (ipc, img_part1.shape[1], v_half_h * 2, v_half_w * 2)
            img_canvas = torch.zeros(img_canvas_shape, device=device)
            img_canvas[:, :, :v_half_h, :v_half_w] = img_part1
            img_canvas[:, :, v_half_h:, :v_half_w] = img_part2
            img_canvas[:, :, :v_half_h, v_half_w:] = img_part3
            img_canvas[:, :, v_half_h:, v_half_w:] = img_part4
            list_syn_img.append(img_canvas)
            
        # 3.5. 为当前类别生成新的标签
        labels_for_class_c = torch.full((ipc,), fill_value=c, dtype=torch.long, device=device)
        list_syn_labels.append(labels_for_class_c)

    # 4. 将所有类别的块拼接成最终的大张量
    print("Concatenating all class buffers...")
    final_aud_syn = torch.cat(list_syn_aud, dim=0) if list_syn_aud else None
    final_image_syn = torch.cat(list_syn_img, dim=0) if list_syn_img else None
    final_labels_syn = torch.cat(list_syn_labels, dim=0) if list_syn_labels else None
    
    # 5. 直接替换“容器”数据集的内部属性
    print("Populating synthetic dataset object...")
    dst_syn_container.audio = final_aud_syn
    dst_syn_container.images = final_image_syn
    dst_syn_container.labels = final_labels_syn
    
    # 6. 更新元数据
    if final_labels_syn is not None:
        dst_syn_container._len = len(final_labels_syn)
    else:
        dst_syn_container._len = 0
    dst_syn_container.num_classes = num_classes
    
    print("Synthetic dataset object is ready.")
    
    # 7. 返回被修改和填充好的容器对象
    return dst_syn_container