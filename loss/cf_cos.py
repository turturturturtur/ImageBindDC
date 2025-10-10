import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import read_cfg
from registry import LOSS
from base import BaseLoss
from .NCFM import match_loss, CFLossFunc


@LOSS.register("cf_cos_loss")
class cf_cos_loss(BaseLoss):
    def __init__(self, **kwargs):
        super(cf_cos_loss, self).__init__()
        loss_cfg = read_cfg("config/loss/cf_cos.yaml")
        kwargs.update(loss_cfg['params'])

        self.lam_base = kwargs.get("lam_base", 1.0)
        self.lam_icm = kwargs.get("lam_icm", 10.0)
        self.lam_cgm = kwargs.get("lam_cgm", 10.0)

        self.alpha_for_loss = kwargs.get("alpha_for_loss")
        self.beta_for_loss = kwargs.get("beta_for_loss")

        self.num_freqs = kwargs.get("num_freqs")

        self.cf_loss = CFLossFunc(
            alpha_for_loss=self.alpha_for_loss, beta_for_loss=self.beta_for_loss
        )

    
    def forward(self, embd_aud_real, embd_aud_syn, embd_img_real, embd_img_syn):
        # 1. 基础分布匹配损失 (Base DM Loss, a.k.a. CF Loss)
        loss_base_aud = match_loss(embd_aud_real, embd_aud_syn, self.cf_loss, self.num_freqs)
        loss_base_vis = match_loss(embd_img_real, embd_img_syn, self.cf_loss, self.num_freqs)
        loss_base_weighted = self.lam_base * (loss_base_aud + loss_base_vis) # 应用权重

        # 2. 隐式交叉匹配损失 (ICM Loss, a.k.a. Cos Loss 1)
        embd_aud_real_norm = F.normalize(embd_aud_real, p=2, dim=1)
        embd_img_real_norm = F.normalize(embd_img_real, p=2, dim=1)
        embd_aud_syn_norm = F.normalize(embd_aud_syn, p=2, dim=1)
        embd_img_syn_norm = F.normalize(embd_img_syn, p=2, dim=1)
        real_combined = embd_aud_real_norm * embd_img_real_norm
        syn_combined = embd_aud_syn_norm * embd_img_syn_norm
        cos_sim = torch.mm(real_combined, syn_combined.T)
        loss_icm_weighted = self.lam_icm * torch.mean(1 - cos_sim) # 应用权重

        # 3. 跨模态间隙匹配损失 (CGM Loss, a.k.a. Cos Loss 2)
        aud_real_avg = torch.mean(embd_aud_real, dim=0, keepdim=True)
        img_syn_avg = torch.mean(embd_img_syn, dim=0, keepdim=True)
        img_real_avg = torch.mean(embd_img_real, dim=0, keepdim=True)
        aud_syn_avg = torch.mean(embd_aud_syn, dim=0, keepdim=True)
        cross_cos_lsss = torch.mm(aud_real_avg * img_syn_avg, (img_real_avg * aud_syn_avg).T)
        loss_cgm_weighted = self.lam_cgm * torch.mean(1 - cross_cos_lsss) # 应用权重

        return loss_base_weighted + loss_icm_weighted + loss_cgm_weighted
    