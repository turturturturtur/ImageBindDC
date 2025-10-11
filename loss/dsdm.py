import torch
import torch.nn as nn
import torch.nn.functional as F
from registry import LOSS
from base import BaseLoss
from utils import read_cfg

@LOSS.register("dsdm_loss")
class FeatureDistributionLoss(BaseLoss):
    """
    特征分布匹配损失函数 (Feature Distribution Matching Loss)。

    该损失函数旨在通过匹配真实数据和合成数据的特征统计量来指导数据蒸馏。
    它包含三个主要部分：
    1. 原型损失 (Prototype Loss): 匹配特征的均值。
    2. 语义损失 (Semantic Loss): 匹配特征的协方差矩阵，以捕捉数据分布的形状。
    3. 历史原型损失 (Historical Prototype Loss): 可选，用于平滑训练过程，将当前合成特征的均值与一个历史移动平均值对齐。
    """
    def __init__(self, **kwargs):
        """
        初始化函数。

        Args:
            metric (str): 用于计算距离的度量方法，可选值为 'mse', 'l1', 'cos'等。默认为 'mse'。
            cov_weight (float): 语义损失（协方差匹配）的权重。默认为 1.0。
            h_p_weight (float): 历史原型损失的权重。默认为 1.0。
        """
        super(FeatureDistributionLoss, self).__init__()

        # 您可以像参考示例一样从配置文件读取参数
        loss_cfg = read_cfg("config/loss/dsdm.yaml")
        kwargs.update(loss_cfg['params'])

        self.metric = kwargs.get("metric")
        self.cov_weight = kwargs.get("cov_weight")
        self.h_p_weight = kwargs.get("h_p_weight")

        print(f"FeatureDistributionLoss initialized with: \n"
              f"  metric='{self.metric}', \n"
              f"  cov_weight={self.cov_weight}, \n"
              f"  h_p_weight={self.h_p_weight}")

    def _dist(self, x, y, method='mse'):
        """计算两个张量之间的距离"""
        if method == 'mse':
            return (x - y).pow(2).sum()
        elif method == 'l1':
            return (x - y).abs().sum()
        elif method == 'cos':
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            # 避免除以零
            return 1 - F.cosine_similarity(x, y, dim=1, eps=1e-6)
        else:
            raise ValueError(f"Unknown distance method: {method}")

    def forward(self, feat_aud_real, feat_aud_syn, feat_real: torch.Tensor, feat_syn: torch.Tensor, h_p: torch.Tensor = None):
        """
        前向传播函数。

        Args:
            feat_real (torch.Tensor): 从真实图像中提取的特征张量，形状为 (N, C, H, W) 或 (N, D)。
            feat_syn (torch.Tensor): 从合成图像中提取的特征张量，形状为 (M, C, H, W) 或 (M, D)。
            h_p (torch.Tensor, optional): 历史原型（特征均值的指数移动平均），用于平滑训练。默认为 None。

        Returns:
            torch.Tensor: 计算出的总损失。
        """
        # --- 1. 原型损失 (均值匹配) ---
        proto_real = feat_real.mean(0)
        proto_syn = feat_syn.mean(0)
        
        proto_loss = self._dist(proto_real, proto_syn, method=self.metric)

        # --- 2. 语义损失 (协方差匹配) ---
        # 将特征图展平以便计算协方差
        feat_real_flat = feat_real.view(feat_real.size(0), -1)
        feat_syn_flat = feat_syn.view(feat_syn.size(0), -1)

        proto_real_flat = feat_real_flat.mean(0)
        proto_syn_flat = feat_syn_flat.mean(0)
        
        # 中心化
        centered_real = feat_real_flat - proto_real_flat
        centered_syn = feat_syn_flat - proto_syn_flat

        # 计算协方差矩阵
        # 使用(N-1)作为分母，是无偏估计
        cov_real = torch.matmul(centered_real.t(), centered_real) / (centered_real.size(0) - 1)
        cov_syn = torch.matmul(centered_syn.t(), centered_syn) / (centered_syn.size(0) - 1)
        
        # 获取特征维度用于归一化
        feature_dim = proto_syn_flat.shape[0]
        
        semantic_loss = self.cov_weight * self._dist(cov_syn, cov_real, method=self.metric) / feature_dim
        
        # --- 合并基础损失 ---
        loss = proto_loss + semantic_loss
        
        # --- 3. 历史原型损失 (可选) ---
        if h_p is not None:
            # 确保h_p和proto_syn的形状一致
            if h_p.shape != proto_syn.shape:
                 raise ValueError(f"Shape mismatch: h_p shape {h_p.shape} != proto_syn shape {proto_syn.shape}")
            
            h_p_loss = self.h_p_weight * self._dist(proto_syn, h_p, method=self.metric) / feature_dim
            loss = loss + h_p_loss

        return loss