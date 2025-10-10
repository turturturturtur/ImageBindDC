import torch
import torch.nn as nn
from registry import LOSS

@LOSS.register("cf_cos_loss")
class cf_cos_loss(nn.Module):
    def __init__(self, **kwargs):
        super(cf_cos_loss, self).__init__()
        self.lam_base = kwargs.get("lam_base", 1.0)
        self.lam_icm = kwargs.get("lam_icm", 10.0)
        self.lam_cgm = kwargs.get("lam_cgm", 10.0)
