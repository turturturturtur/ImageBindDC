import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import read_cfg
from registry import LOSS
from base import BaseLoss
from .NCFM import match_loss, CFLossFunc


@LOSS.register("cf_cos_loss_vision")
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
        loss_base_vis = match_loss(embd_img_real, embd_img_syn, self.cf_loss, self.num_freqs)
        return loss_base_vis * self.lam_base
    