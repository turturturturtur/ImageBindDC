import torch.nn as nn
import torch
class Classifier_Ensemble(nn.Module):
    def __init__(self, cls_num, input_modality, input_size=6272):
        super(Classifier_Ensemble, self).__init__()
        dim_size = 6272*2 #20608
        if input_modality == 'v':
            dim_size = 6272#512*7*7
        elif input_modality == 'a':
            dim_size = 2048
        self.fc_a = nn.Linear(input_size, cls_num)
        self.fc_v = nn.Linear(input_size, cls_num)

    def forward(self, feat_sound, feat_img):
        g_a, g_v = None, None
        if feat_sound is not None:
            g_a = self.fc_a(feat_sound)
        if feat_img is not None:
            g_v = self.fc_v(feat_img)
        if g_a is None and g_v is not None:
            g_a = torch.zeros_like(g_v)
        if g_v is None and g_a is not None:
            g_v = torch.zeros_like(g_a)

        return g_a, g_v