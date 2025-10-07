


# # Numerical libs
# import torch
# import torch.nn.functional as F
# from nets.imagebind import data
# from nets.imagebind.models.imagebind_model import ModalityType, ImageBindModel

# # Network wrapper, defines forward pass
# class NetWrapper(torch.nn.Module):
#     def __init__(self, args, nets):
#         super(NetWrapper, self).__init__() 
#         self.net_sound, self.net_frame, self.net_classifier = nets
#         self.args = args
    
#     #def forward(self, audio, frame):
#     #    feat_sound, feat_frame = None, None
#     #    if self.net_sound is not None:
#     #        feat_sound = self.net_sound(audio)
#     #    if self.net_frame is not None:
#     #        feat_frame = self.net_frame(frame)
#     #    pred = self.net_classifier(feat_sound, feat_frame)
#     #    return pred
#     def forward(self, audio, frame):
#         feat_sound, feat_frame = None, None

#         # 音频
#         if self.net_sound is not None:
#             if isinstance(self.net_sound, ImageBindModel):
#                 audio = F.interpolate(audio, size=(128, 204), mode='bilinear', align_corners=False)
#                 inputs = {ModalityType.AUDIO: audio}
#                 feat_sound = self.net_sound(inputs)[ModalityType.AUDIO]
#             else:
#                 feat_sound = self.net_sound(audio)

#         # 视觉
#         if self.net_frame is not None:
#             if isinstance(self.net_frame, ImageBindModel):
#                 frame = F.interpolate(frame, size=(224, 224), mode='bilinear', align_corners=False)
#                 inputs = {ModalityType.VISION: frame}
#                 feat_frame = self.net_frame(inputs)[ModalityType.VISION]
#             else:
#                 feat_frame = self.net_frame(frame)

#         # 拼接特征
#         if feat_sound is not None and feat_frame is not None:
#             feat = torch.cat([feat_sound, feat_frame], dim=1)
#         elif feat_sound is not None:
#             feat = feat_sound
#         elif feat_frame is not None:
#             feat = feat_frame
#         else:
#             raise ValueError("Both audio and frame features are None")

#         pred = self.net_classifier(feat)
#         return pred

#     def get_embds(self, audio, frame):
#         feat_sound, feat_frame = None, None
#         if self.net_sound is not None:
#             feat_sound = self.net_sound(audio)
#         if self.net_frame is not None:
#             feat_frame = self.net_frame(frame)
#         return feat_sound, feat_frame


# class MTTNetWrapper(torch.nn.Module):
#     """
#     专门为MTT方法设计的网络包装器
#     """
#     def __init__(self, args, nets):
#         super(MTTNetWrapper, self).__init__()
#         self.net_sound, self.net_frame, self.net_classifier = nets[0], nets[1], nets[2]
#         self.arch_classifier = args.arch_classifier

#     def forward(self, audio, frame, flat_param=None):
#         # ReparamModule 的机制是：如果提供了flat_param，它会在内部自动用这些参数
#         # 来覆盖模型的原生参数，然后再执行下面的原始逻辑。
#         # 因此，我们不需要在这里手动处理 flat_param，只需确保函数签名中有它即可。

#         embd_a = self.net_sound.embed(audio) if audio is not None and self.net_sound is not None else None
#         embd_v = self.net_frame.embed(frame) if frame is not None and self.net_frame is not None else None

#         logits_a, logits_v = self.net_classifier(embd_a, embd_v)

#         if self.arch_classifier == 'ensemble':
#             return logits_a, logits_v
#         else:
  
#             if logits_a is not None and logits_v is not None:

#                 return (logits_a + logits_v) / 2
#             elif logits_a is not None:
 
#                 return logits_a
#             elif logits_v is not None:

#                 return logits_v
#             else:
#                 return None

# # class NetWrapperimagebind(torch.nn.Module):
# #     def __init__(self, args, nets):
# #         super(NetWrapperimagebind, self).__init__() 
# #         self.net_imagebind, self.net_classifier = nets
# #         self.args = args

# #     def forward(self, audio, frame):
# #         feat_sound, feat_frame = None, None
# #         # audio = F.interpolate(audio, size=(128, 204), mode='bilinear', align_corners=False)
# #         # audio = audio.unsqueeze(1).repeat(1, 3, 1, 1, 1)
# #         # if self.net_sound is not None:
# #         #     feat_sound = self.net_sound(audio)
# #         # if self.net_frame is not None:
# #         #     feat_frame = self.net_frame(frame)
# #         inputs = {
# #             ModalityType.VISION: frame, #(1, 3, 224, 224)
# #             ModalityType.AUDIO: audio}
# #         embeddings = self.net_imagebind(inputs)
# #         feat_frame, feat_sound = embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO]
        
# #         pred = self.net_classifier(feat_sound, feat_frame)
# #         return pred

# #     def get_embds(self, audio, frame):
# #         feat_sound, feat_frame = None, None
# #         # audio = F.interpolate(audio, size=(128, 204), mode='bilinear', align_corners=False)
# #         # audio = audio.unsqueeze(1).repeat(1, 3, 1, 1, 1)
# #         # if self.net_sound is not None:
# #         #     feat_sound = self.net_sound(audio)
# #         # if self.net_frame is not None:
# #         #     feat_frame = self.net_frame(frame)
# #         inputs = {
# #             ModalityType.VISION: frame, #(1, 3, 224, 224)
# #             ModalityType.AUDIO: audio}
# #         embeddings = self.net_imagebind(inputs)
# #         feat_frame, feat_sound = embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO]
# #         return feat_sound, feat_frame

# def adjust_learning_rate(optimizer, args):
#     args.lr_sound *= 0.1
#     args.lr_frame *= 0.1
#     args.lr_classifier *= 0.1
#     for param_group in optimizer.param_groups:
#         param_group['lr'] *= 0.1

# # class NetWrapperimagebind(torch.nn.Module):
# #     def __init__(self, args, nets):
# #         super().__init__()
# #         self.net_imagebind, self.net_classifier = nets
# #         self.args = args

# #     def forward(self, audio, frame):
# #         inputs = {
# #             ModalityType.AUDIO: audio,   # [B, 1, 128, 204]
# #             ModalityType.VISION: frame   # [B, 3, 224, 224]
# #         }
# #         embeddings = self.net_imagebind(inputs)
# #         feat_sound = embeddings[ModalityType.AUDIO]  # [B, 1024]
# #         feat_frame = embeddings[ModalityType.VISION] # [B, 1024]

# #         # 融合：简单拼接（可替换成加权、注意力等）
# #         feat = torch.cat([feat_sound, feat_frame], dim=1)  # [B, 2048]
# #         pred = self.net_classifier(feat)                   # [B, num_classes]
# #         return pred

# #     def get_embds(self, audio, frame):
# #         inputs = {
# #             ModalityType.AUDIO: audio,
# #             ModalityType.VISION: frame
# #         }
# #         embeddings = self.net_imagebind(inputs)
# #         return embeddings[ModalityType.AUDIO], embeddings[ModalityType.VISION]

# import torch
# import torch.nn.functional as F
# from nets.imagebind.models.imagebind_model import ModalityType

# # 放在 utils/train_utils.py 末尾

# class NetWrapperimagebind(torch.nn.Module):


#     def forward(self, audio, frame):
#         # 添加音频输入尺寸校正（内存规范1要求128x204）
#         # 修正：保持原始尺寸传递逻辑，仅在必要时插值（保持向后兼容）
#         if audio.dim() == 3:  # 处理输入维度不一致情况
#             audio = audio.unsqueeze(1)  # [B, 1, 128, 204]
            
#         feat_sound, feat_frame = self.get_embds(audio, frame)

#         # 添加维度校验断言（符合内存规范第6条）
#         if feat_sound is not None and feat_sound.dim() == 2:
#             assert feat_sound.size(1) == 1024, f"音频特征维度错误: {feat_sound.size(1)}"
#         if feat_frame is not None and feat_frame.dim() == 2:
#             assert feat_frame.size(1) == 1024, f"视觉特征维度错误: {feat_frame.size(1)}"

#         # 拼接后分类 (逻辑保持不变)
#         if feat_sound is not None and feat_frame is not None:
#             feat = torch.cat([feat_sound, feat_frame], dim=1)  # 1024 + 1024 = 2048
#         elif feat_sound is not None:
#             # 添加单模态断言
#             assert feat_sound.size(1) == 1024, "单模态音频特征维度错误"
#             feat = feat_sound
#         else:
#             # 添加单模态断言
#             assert feat_frame.size(1) == 1024, "单模态视觉特征维度错误"
#             feat = feat_frame
            
#         pred = self.net_classifier(feat)
#         return pred

# # 修正分类器输入维度为2048，保持与特征拼接结果匹配


# Numerical libs
import torch

import torch.nn as nn
import torch.nn.functional as F
from nets.imagebind.models.imagebind_model import ModalityType
from nets.imagebind import data
# Network wrapper, defines forward pass
# in utils/train_utils.py

# ... (确保文件顶部有 import torch.nn as nn) ...
# in utils/train_utils.py
import torch
import torch.nn as nn

class NetWrapper(nn.Module):
    """
    一个通用的、稳健的网络包装器，能够处理单/多GPU情况。
    """
    def __init__(self, args, nets):
        super(NetWrapper, self).__init__() 
        # 确保nets参数正确解包
        if nets is None or len(nets) != 2:
            raise ValueError("nets参数必须包含net_imagebind和net_classifier两个元素")
        self.net_imagebind, self.net_classifier = nets
        self.args = args
    def forward(self, audio, frame):
        # 注意：这里我们不再调用 .embed，而是直接调用子网络本身
        # 这要求子网络（如ConvNet）的forward方法返回的就是特征
        feat_sound = self.net_sound(audio) if audio is not None else None
        feat_frame = self.net_frame(frame) if frame is not None else None
        pred = self.net_classifier(feat_sound, feat_frame)
        return pred
        
    def get_embds(self, audio, frame):
        """这个方法专门负责提取中间层的特征嵌入"""
        feat_sound, feat_frame = None, None
        
        # [核心逻辑] 安全地调用embed方法，处理DataParallel包装
        if audio is not None and self.net_sound is not None:
            if isinstance(self.net_sound, nn.DataParallel):
                feat_sound = self.net_sound.module.embed(audio)
            else:
                feat_sound = self.net_sound.embed(audio)

        if frame is not None and self.net_frame is not None:
            if isinstance(self.net_frame, nn.DataParallel):
                feat_frame = self.net_frame.module.embed(frame)
            else:
                feat_frame = self.net_frame.embed(frame)
                
        return feat_sound, feat_frame

# class NetWrapper(torch.nn.Module):
#     def __init__(self, args, nets):
#         super(NetWrapper, self).__init__() 
#         self.net_sound, self.net_frame, self.net_classifier = nets
#         self.args = args

#     def forward(self, audio, frame):
#         feat_sound, feat_frame = None, None
#         if self.net_sound is not None:
#             feat_sound = self.net_sound(audio)
#         if self.net_frame is not None:
#             feat_frame = self.net_frame(frame)
#         pred = self.net_classifier(feat_sound, feat_frame)
#         return pred

#     def get_embds(self, audio, frame):
#         feat_sound, feat_frame = None, None
#         if self.net_sound is not None:
#             feat_sound = self.net_sound(audio)
#         if self.net_frame is not None:
#             feat_frame = self.net_frame(frame)
#         return feat_sound, feat_frame

class MTTNetWrapper(torch.nn.Module):
    """
    专为 MTT 设计的网络包装器 —— 不再使用任何投影层
    """
    def __init__(self, args, nets):
        super().__init__()
        self.net_sound, self.net_frame, self.net_classifier = nets[0], nets[1], nets[2]
        self.arch_classifier = args.arch_classifier

    def forward(self, audio, frame, flat_param=None):
        embd_a = (
            self.net_sound.module.embed(audio)
            if audio is not None and isinstance(self.net_sound, torch.nn.DataParallel)
            else self.net_sound.embed(audio)
            if audio is not None
            else None
        )
        embd_v = (
            self.net_frame.module.embed(frame)
            if frame is not None and isinstance(self.net_frame, torch.nn.DataParallel)
            else self.net_frame.embed(frame)
            if frame is not None
            else None
        )

        logits_a, logits_v = self.net_classifier(embd_a, embd_v)

        if self.arch_classifier == 'ensemble':
            return logits_a, logits_v
        else:
            if logits_a is not None and logits_v is not None:
                return (logits_a + logits_v) / 2
            elif logits_a is not None:
                return logits_a
            elif logits_v is not None:
                return logits_v
            else:
                return None


class NetWrapperimagebind(torch.nn.Module):
    def __init__(self, args, nets):
        super(NetWrapperimagebind, self).__init__()
        self.net_imagebind, self.net_classifier = nets
        self.args = args

    def forward(self, audio, frame):
        # 允许单模态缺失
        inputs = {}
        if audio is not None:
            inputs[ModalityType.AUDIO] = audio
        if frame is not None:
            inputs[ModalityType.VISION] = frame

        embeddings = self.net_imagebind(inputs)
        feat_sound = embeddings.get(ModalityType.AUDIO)
        feat_frame = embeddings.get(ModalityType.VISION)

        # 拼接或单模态
        if feat_sound is not None and feat_frame is not None:
            feat = feat_sound + feat_frame
        elif feat_sound is not None:
            feat = feat_sound
        elif feat_frame is not None:
            feat = feat_frame
        else:
            raise ValueError("Both audio and frame are None")

        pred = self.net_classifier(feat)
        return pred

    def get_embds(self, audio, frame):
        inputs = {}
        if audio is not None:
            inputs[ModalityType.AUDIO] = audio
        if frame is not None:
            inputs[ModalityType.VISION] = frame

        embeddings = self.net_imagebind(inputs)
        feat_sound = embeddings.get(ModalityType.AUDIO)
        feat_frame = embeddings.get(ModalityType.VISION)
        return feat_sound, feat_frame

def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_classifier *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

def get_network_imagebind(args, pretrained=True):
    from nets import ModelBuilder
    from utils.train_utils import NetWrapperimagebind

    builder = ModelBuilder()

    net_imagebind = builder.build_imagebind(arch=args.arch_frame, pretrained=pretrained)
    
    # 明确标注input_size修改位置：2048=音频1024+视觉1024（内存规范6）
    net_classifier = builder.build_classifier(
        arch=args.arch_classifier,
        cls_num=args.cls_num,
        weights=args.weights_classifier,
        input_modality=args.input_modality,
        input_size=1024)  # 关键参数修改
        
    # 添加维度校验钩子函数（内存规范6）
    def check_input_dim(module, input):
        assert input[0].size(1) == 1024, f"分类器输入维度错误: {input[0].size(1)}"
    net_classifier.register_forward_pre_hook(check_input_dim)
    
    nets = (net_imagebind, net_classifier)          
    netWrapper = NetWrapperimagebind(args, nets)
    return nets, netWrapper
