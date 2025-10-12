import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import read_cfg
from registry import LOSS
from base import BaseLoss
from .NCFM import match_loss, CFLossFunc


@LOSS.register("dm_loss")
class cf_cos_loss(BaseLoss):
    def __init__(self, **kwargs):
        super(cf_cos_loss, self).__init__()

    
    def forward(self, embd_aud_real, embd_aud_syn, embd_img_real, embd_img_syn):
        """
        计算真实嵌入和合成嵌入之间的L2损失。

        参数:
            embd_aud_real (Tensor): 真实的音频嵌入。
            embd_aud_syn (Tensor): 合成的音频嵌入。
            embd_img_real (Tensor): 真实的图像嵌入。
            embd_img_syn (Tensor): 合成的图像嵌入。

        返回:
            Tensor: 真实损失和合成损失之和。
        """
        # 计算真实音频和图像嵌入之间的L2损失（均方误差）
        loss_real = F.mse_loss(embd_aud_real, embd_img_real)
        
        # 计算合成音频和图像嵌入之间的L2损失（均方误差）
        loss_syn = F.mse_loss(embd_aud_syn, embd_img_syn)
        
        # 将两个损失直接相加
        total_loss = loss_real + loss_syn
        
        return total_loss
    