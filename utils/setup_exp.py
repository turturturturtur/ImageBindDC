from torchvision.transforms import v2
import torch
from tqdm import tqdm
import numpy as np
from .class_sample import ClassSampler, downscale


def get_img_transform():
    img_transform = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ])
    return img_transform

def get_syn_data(dst_train, ipc):
    """
    通过对每个类别进行采样、缩放和拼接，动态构建合成数据集。

    Args:
        dst_train (Dataset): 包含所有真实数据的源数据集。
        ipc (int): Images Per Class，每个类别最终生成的合成样本数量。

    Returns:
        Tuple[Tensor, Tensor]: 包含合成音频和图像数据的大张量。
    """
    # 1. 初始化 ClassSampler
    sampler = ClassSampler(dataset=dst_train)
    num_classes = dst_train.num_classes
    device = dst_train.audio.device if dst_train.audio is not None else dst_train.images.device

    # 2. 创建空列表来收集每个类别处理好的“四合一”数据块
    list_syn_aud = []
    list_syn_img = []

    print("Dynamically building synthetic data buffers...")
    for c in tqdm(range(num_classes), desc="Processing Classes"):
        # 3.1. 计算本轮需要采样的样本总数 (仍然是 ipc * 4)
        num_samples_to_get = ipc * 4
        
        # 3.2. 从 sampler 获取随机索引
        selected_indices = sampler.sample(class_idx=c, num_samples=num_samples_to_get)
        
        if not selected_indices:
            print(f"Warning: No samples found for class {c}. Creating zero tensors.")
            # 如果某个类没有样本，为了保持最终张量的形状，我们需要添加一个零张量
            # 你需要在这里定义零张量的 shape
            # aud_zero_shape = (ipc, C_a, 1, H_a, W_a) # 示例
            # img_zero_shape = (ipc, C_v, H_v, W_v)   # 示例
            # list_syn_aud.append(torch.zeros(aud_zero_shape, device=device))
            # list_syn_img.append(torch.zeros(img_zero_shape, device=device))
            continue
            
        # 3.3. 批量从数据集中取出真实数据
        batch_data = dst_train[selected_indices]
        auds_real = batch_data['audio'].to(device)
        imgs_real = batch_data['frame'].to(device)
        
        # 进行必要的形状调整
        if auds_real.dim() == 4: auds_real = auds_real.unsqueeze(2)
        if auds_real.dim() == 3: auds_real = auds_real.unsqueeze(1).unsqueeze(1)

        # 3.4. 【核心修改】为当前类别构建“四合一”数据块
        # --- 音频部分 ---
        if auds_real is not None:
            # 从采样数据中切片出4个部分
            aud_part1 = downscale(auds_real[0*ipc:1*ipc].squeeze(2), 0.5).unsqueeze(2)
            aud_part2 = downscale(auds_real[1*ipc:2*ipc].squeeze(2), 0.5).unsqueeze(2)
            aud_part3 = downscale(auds_real[2*ipc:3*ipc].squeeze(2), 0.5).unsqueeze(2)
            aud_part4 = downscale(auds_real[3*ipc:4*ipc].squeeze(2), 0.5).unsqueeze(2)

            # 获取下采样后的小块尺寸
            _, _, _, a_half_h, a_half_w = aud_part1.shape
            
            # 创建一个当前类别大小的零张量作为画布
            aud_canvas_shape = (ipc, aud_part1.shape[1], 1, a_half_h * 2, a_half_w * 2)
            aud_canvas = torch.zeros(aud_canvas_shape, device=device)
            
            # 将4个小块拼接到画布上
            aud_canvas[:, :, :, :a_half_h, :a_half_w] = aud_part1
            aud_canvas[:, :, :, a_half_h:, :a_half_w] = aud_part2
            aud_canvas[:, :, :, :a_half_h, a_half_w:] = aud_part3
            aud_canvas[:, :, :, a_half_h:, a_half_w:] = aud_part4
            
            list_syn_aud.append(aud_canvas)

        # --- 图像部分 ---
        if imgs_real is not None:
            # 从采样数据中切片出4个部分
            img_part1 = downscale(imgs_real[0*ipc:1*ipc], 0.5)
            img_part2 = downscale(imgs_real[1*ipc:2*ipc], 0.5)
            img_part3 = downscale(imgs_real[2*ipc:3*ipc], 0.5)
            img_part4 = downscale(imgs_real[3*ipc:4*ipc], 0.5)
            
            # 获取下采样后的小块尺寸
            _, _, v_half_h, v_half_w = img_part1.shape

            # 创建画布
            img_canvas_shape = (ipc, img_part1.shape[1], v_half_h * 2, v_half_w * 2)
            img_canvas = torch.zeros(img_canvas_shape, device=device)
            
            # 拼接
            img_canvas[:, :, :v_half_h, :v_half_w] = img_part1
            img_canvas[:, :, v_half_h:, :v_half_w] = img_part2
            img_canvas[:, :, :v_half_h, v_half_w:] = img_part3
            img_canvas[:, :, v_half_h:, v_half_w:] = img_part4 # 修正了之前代码中的一个小 typo (v_half_w)
            
            list_syn_img.append(img_canvas)

    # 4. 【最终步骤】将列表中所有类别的“四合一”数据块拼接成最终的大张量
    print("Concatenating all class buffers...")
    final_aud_syn = torch.cat(list_syn_aud, dim=0) if list_syn_aud else None
    final_image_syn = torch.cat(list_syn_img, dim=0) if list_syn_img else None
    
    print("Synthetic data buffers built successfully.")
    
    return final_aud_syn, final_image_syn