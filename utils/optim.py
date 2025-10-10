import torch

def get_syn_optimizer(dst_syn, input_modality, exp_cfg):
    param_groups = []
    if input_modality == 'a' or input_modality == 'av':
        param_groups += [{'params': dst_syn.audio, 'lr': exp_cfg.get("lr_syn_aud", 0.1)}]
    
    if input_modality == 'v' or input_modality == 'av':
        param_groups += [{'params': dst_syn.images, 'lr': exp_cfg.get("lr_syn_img", 0.1)}]

    if exp_cfg.get("soft_label") != False:
        param_groups += [{'params': dst_syn.labels, 'lr': exp_cfg.get("lr_syn_label", 0.1)}]
    return torch.optim.SGD(param_groups, momentum=0.5)