import clip
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union
from registry import MODEL
from torchvision.transforms import Normalize
from .convnet import ConvNet


@MODEL.register("clip_linear")
class ClipClassifier_linear(nn.Module):
    def __init__(self,**kwargs):
        extra_params = kwargs.pop('extra_params', None)
        super(ClipClassifier_linear, self).__init__()
        self.CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
        self.CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _ = clip.load("ViT-B/32", device=self.device, download_root="data/checkpoint")

        self.classifier_image = nn.Linear(extra_params.get("input_dim"), extra_params.get("num_classes"))
        self.classifier_audio = nn.Linear(extra_params.get("input_dim"), extra_params.get("num_classes"))

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
        

@MODEL.register("clip_mlp")
class ClipClassifier_mlp(nn.Module):
    def __init__(self,**kwargs):
        extra_params = kwargs.pop('extra_params', None)
        super(ClipClassifier_mlp, self).__init__()
        self.normalize = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _ = clip.load("ViT-B/32", device=self.device, download_root="data/checkpoint")

        input_dim = extra_params.get("input_dim", 512) 
        num_classes = extra_params.get("num_classes")
        hidden_dim = extra_params.get("hidden_dim", 256) # MLP 隐藏层维度
        dropout_p = extra_params.get("dropout_p", 0.5)   # Dropout 概率

        self.classifier_image = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_classes)
        )
        self.classifier_audio = nn.Linear(extra_params.get("input_dim"), extra_params.get("num_classes"))

    def forward(self, inputs, mode: Optional[str] = None):
        '''
        单一模态，只能处理inputs了
        '''
        image = inputs["image"].to(self.device)
        image = self.normalize(image)
        
        # clip_model.encode_image 会返回 [N, 512] 的张量
        features = self.model.encode_image(image).to(torch.float32)
            
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
@MODEL.register("clip_convnet")
class ClipClassifier_convnet(nn.Module):
    def __init__(self,**kwargs):
        extra_params = kwargs.pop('extra_params', None)
        super(ClipClassifier_convnet, self).__init__()
        self.normalize = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _ = clip.load("ViT-B/32", device=self.device, download_root="data/checkpoint")

        num_classes = extra_params.get("num_classes")
        pseudo_im_size = (16, 32)
        in_channels = 1

        print(f"Initializing ConvNet classifier with input shape (1, 16, 32) and mode='audio'...")
        self.classifier_image = ConvNet(
            channel=in_channels, 
            im_size=pseudo_im_size, 
            mode='audio', # 使用 audio 模式以利用自适应池化
            num_classes=num_classes
        )
        self.classifier_audio = nn.Linear(extra_params.get("input_dim"), extra_params.get("num_classes"))

    def forward(self, inputs, mode: Optional[str] = None):
        '''
        单一模态，只能处理inputs了
        '''
        image = inputs["image"].to(self.device)
        image = self.normalize(image)

        # clip_model.encode_image 会返回 [N, 512] 的张量
        features = self.model.encode_image(image).to(torch.float32)

        # 如果需要embeddings就直接返回
        if mode == "embeddings":
            return {
                "vision": features,
                "audio": features,

            }

        pseudo_image = features.view(-1, 1, 16, 32)
        logits_image = self.classifier_image(pseudo_image)

        if mode == "logits":
            return logits_image
        
        pred = F.softmax(logits_image, dim=-1)
        return pred