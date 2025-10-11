import clip
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union
from registry import MODEL
from torchvision.transforms import Normalize


@MODEL.register("clip")
class ClipClassifier(nn.Module):
    def __init__(self,**kwargs):
        extra_params = kwargs.pop('extra_params', None)
        super(ClipClassifier, self).__init__()
        self.CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
        self.CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _ = clip.load("ViT-B/32", device=self.device)

        self.classifier_image = nn.Linear(extra_params.get("input_dim"), extra_params.get("num_classes"))

    def forward(self, inputs, mode: Optional[str] = None):
        '''
        单一模态，只能处理inputs了
        '''
        image = inputs["image"].to(self.device)
        image = Normalize(self.CLIP_MEAN, self.CLIP_STD)(image)


        features = self.model.encode_image(image).float()
        # 如果需要embeddings就直接返回
        if mode == "embeddings":
            return {
                "vision": features,
                "audio": features,

            }

        logits_image = self.classifier_image(features)
        if mode == "logits":
            return logits_image
        
        pred = F.softmax(logits_image, dim=-1)
        return pred
        