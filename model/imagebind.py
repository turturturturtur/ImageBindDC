#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from functools import partial
from types import SimpleNamespace
from typing import Optional
import torch
import torch.nn as nn

from .helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                            SelectElement, SelectEOSAndProject)
from .multimodal_preprocessors import (AudioPreprocessor,
                                             IMUPreprocessor, PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper,
                                             TextPreprocessor,
                                             ThermalPreprocessor)
from .transformer import MultiheadAttention, SimpleTransformer
from registry import MODEL
import torch.nn.functional as F


ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)




class ImageBindModel(nn.Module):
    def __init__(
        self,
        video_frames=2,
        kernel_size=(2, 14, 14),
        audio_kernel_size=16,
        audio_stride=10,
        out_embed_dim=768,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_num_mel_bins=128,
        audio_target_len=204,
        audio_drop_path=0.1,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        depth_embed_dim=384,
        depth_kernel_size=16,
        depth_num_blocks=12,
        depth_num_heads=8,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_kernel_size=8,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
    ):
        super().__init__()

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            vision_embed_dim,
            kernel_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
            depth_embed_dim,
            depth_kernel_size,
            thermal_embed_dim,
            thermal_kernel_size,
            imu_embed_dim,
        )

        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_drop_path,
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_drop_path,
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_drop_path,
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_drop_path,
        )

        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            vision_embed_dim,
            text_embed_dim,
            audio_embed_dim,
            depth_embed_dim,
            thermal_embed_dim,
            imu_embed_dim,
        )

        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )

    def _create_modality_preprocessors(
        self,
        video_frames=2,
        vision_embed_dim=1024,
        kernel_size=(2, 14, 14),
        text_embed_dim=768,
        audio_embed_dim=768,
        audio_kernel_size=16,
        audio_stride=10,
        audio_num_mel_bins=128,
        audio_target_len=204,
        depth_embed_dim=768,
        depth_kernel_size=16,
        thermal_embed_dim=768,
        thermal_kernel_size=16,
        imu_embed_dim=512,
    ):
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=kernel_size,
                    bias=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )

        # text_preprocessor = TextPreprocessor(
        #     context_length=77,
        #     vocab_size=49408,
        #     embed_dim=text_embed_dim,
        #     causal_masking=True,
        # )

        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )

        # depth_stem = PatchEmbedGeneric(
        #     [
        #         nn.Conv2d(
        #             kernel_size=depth_kernel_size,
        #             in_channels=1,
        #             out_channels=depth_embed_dim,
        #             stride=depth_kernel_size,
        #             bias=False,
        #         ),
        #     ],
        #     norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
        # )

        # depth_preprocessor = RGBDTPreprocessor(
        #     img_size=[1, 224, 224],
        #     num_cls_tokens=1,
        #     pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        #     rgbt_stem=None,
        #     depth_stem=depth_stem,
        # )

        # thermal_stem = PatchEmbedGeneric(
        #     [
        #         nn.Conv2d(
        #             kernel_size=thermal_kernel_size,
        #             in_channels=1,
        #             out_channels=thermal_embed_dim,
        #             stride=thermal_kernel_size,
        #             bias=False,
        #         ),
        #     ],
        #     norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
        # )
        # thermal_preprocessor = ThermalPreprocessor(
        #     img_size=[1, 224, 224],
        #     num_cls_tokens=1,
        #     pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        #     thermal_stem=thermal_stem,
        # )

        # imu_stem = PatchEmbedGeneric(
        #     [
        #         nn.Linear(
        #             in_features=48,
        #             out_features=imu_embed_dim,
        #             bias=False,
        #         ),
        #     ],
        #     norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
        # )

        # imu_preprocessor = IMUPreprocessor(
        #     img_size=[6, 2000],
        #     num_cls_tokens=1,
        #     kernel_size=8,
        #     embed_dim=imu_embed_dim,
        #     pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
        #     imu_stem=imu_stem,
        # )

        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
        self,
        vision_embed_dim=1024,
        vision_num_blocks=24,
        vision_num_heads=16,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_drop_path=0.0,
        depth_embed_dim=768,
        depth_num_blocks=12,
        depth_num_heads=12,
        depth_drop_path=0.0,
        thermal_embed_dim=768,
        thermal_num_blocks=12,
        thermal_num_heads=12,
        thermal_drop_path=0.0,
        imu_embed_dim=512,
        imu_num_blocks=6,
        imu_num_heads=8,
        imu_drop_path=0.7,
    ):
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )
        # modality_trunks[ModalityType.TEXT] = instantiate_trunk(
        #     text_embed_dim,
        #     text_num_blocks,
        #     text_num_heads,
        #     pre_transformer_ln=False,
        #     add_bias_kv=False,
        #     drop_path=0.0,
        # )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=audio_drop_path,
        )
        # modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
        #     depth_embed_dim,
        #     depth_num_blocks,
        #     depth_num_heads,
        #     pre_transformer_ln=False,
        #     add_bias_kv=True,
        #     drop_path=depth_drop_path,
        # )
        # modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
        #     thermal_embed_dim,
        #     thermal_num_blocks,
        #     thermal_num_heads,
        #     pre_transformer_ln=False,
        #     add_bias_kv=True,
        #     drop_path=thermal_drop_path,
        # )
        # modality_trunks[ModalityType.IMU] = instantiate_trunk(
        #     imu_embed_dim,
        #     imu_num_blocks,
        #     imu_num_heads,
        #     pre_transformer_ln=False,
        #     add_bias_kv=True,
        #     drop_path=imu_drop_path,
        # )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
        self,
        out_embed_dim,
        vision_embed_dim,
        text_embed_dim,
        audio_embed_dim,
        depth_embed_dim,
        thermal_embed_dim,
        imu_embed_dim,
    ):
        modality_heads = {}

        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        # modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
        #     proj=nn.Sequential(
        #         nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
        #         nn.Linear(text_embed_dim, out_embed_dim, bias=False),
        #     )
        # )

        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )

        # modality_heads[ModalityType.DEPTH] = nn.Sequential(
        #     nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
        #     SelectElement(index=0),
        #     nn.Linear(depth_embed_dim, out_embed_dim, bias=False),
        # )

        # modality_heads[ModalityType.THERMAL] = nn.Sequential(
        #     nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
        #     SelectElement(index=0),
        #     nn.Linear(thermal_embed_dim, out_embed_dim, bias=False),
        # )

        # modality_heads[ModalityType.IMU] = nn.Sequential(
        #     nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
        #     SelectElement(index=0),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
        # )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}

        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        # modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
        #     Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        # )
        modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )
        # modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(
        #     Normalize(dim=-1),
        #     LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        # )
        # modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(
        #     Normalize(dim=-1),
        #     LearnableLogitScaling(logit_scale_init=10.0, learnable=False),
        # )
        # modality_postprocessors[ModalityType.IMU] = nn.Sequential(
        #     Normalize(dim=-1),
        #     LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        # )

        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs):
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (
                modality_value.ndim >= 5
            )  # Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                modality_value = self.modality_heads[modality_key](
                    modality_value, **head_inputs
                )
                modality_value = self.modality_postprocessors[modality_key](
                    modality_value
                )

                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(dim=1)

                outputs[modality_key] = modality_value

        return outputs
    
    def embed(self, inputs):
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (
                modality_value.ndim >= 5
            )  # Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                modality_value = self.modality_heads[modality_key](
                    modality_value, **head_inputs
                )
                modality_value = self.modality_postprocessors[modality_key](
                    modality_value
                )

                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(dim=1)

                outputs[modality_key] = modality_value

        return outputs

def audio_padding(audio):
    h, w = audio.shape[-2], audio.shape[-1]
    if h != w:
        # --- 开始替换 F.interpolate 逻辑 ---
        TARGET_WIDTH = 204
        
        if w < TARGET_WIDTH:
            # 如果当前宽度小于目标宽度，则在右侧填充
            padding_needed = TARGET_WIDTH - w
            audio = F.pad(audio, (0, padding_needed), "constant", 0)
        elif w > TARGET_WIDTH:
            # 如果当前宽度大于目标宽度，则从右侧截断
            audio = audio[..., :TARGET_WIDTH]
        else:
            # 如果宽度正好等于目标宽度，则无需操作
            audio = audio
    return audio

@MODEL.register("imagebind")
class ImageBindClassifer(ImageBindModel):
    def __init__(self,**kwargs):
        extra_params = kwargs.pop('extra_params', None)
        super(ImageBindClassifer, self).__init__(**kwargs)

        # 定义权重文件的路径
        pretrained_path = "data/checkpoint/imagebind_huge.pth"
        
        # 检查权重文件是否存在，如果不存在则自动下载
        if not os.path.exists(pretrained_path):
            print(f"ImageBind weights not found at '{pretrained_path}'. Downloading...")
            
            # 确保 checkpoint 目录存在
            checkpoint_dir = os.path.dirname(pretrained_path)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 从官方 URL 下载文件
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                pretrained_path,
                progress=True,
            )
        
        # 加载预训练权重到模型中
        print(f"Loading pretrained ImageBind weights from: {pretrained_path}")
        # 使用 strict=False 允许我们只加载父类 ImageBindModel 的权重，
        # 而忽略我们自己新加的 classifier 层的权重（因为它们在 .pth 文件中不存在）。
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        print("Pretrained weights loaded successfully.")

        self.classifier_audio = nn.Linear(extra_params.get("input_dim"), extra_params.get("num_classes"))
        self.classifier_image = nn.Linear(extra_params.get("input_dim"), extra_params.get("num_classes"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def forward(self, inputs, mode: Optional[str] = None):
        '''
        直接输入dst_train形式的inputs,返回分类结果
        '''
        # 自动获取模型当前设备
        device = next(self.parameters()).device
        
        audio = inputs["audio"].to(device)
        image = inputs["image"].to(device)

        # 填充audio
        audio = audio_padding(audio)

        inputs = {
            ModalityType.VISION: image,
            ModalityType.AUDIO: audio,
        }
        features = super(ImageBindClassifer, self).forward(inputs)
        
        # 如果需要embeddings就直接返回
        if mode == "embeddings":
            return features
        
        # 分解模态feature准备送入classifier
        if ModalityType.VISION in features:
            feature_image = features[ModalityType.VISION]
            logits_image = self.classifier_image(feature_image)
        if ModalityType.AUDIO in features:
            feature_audio = features[ModalityType.AUDIO]
            logits_audio = self.classifier_audio(feature_audio)
        else:
            raise ValueError("No valid modality found for classification.")

        if mode == "logits":
            return logits_image, logits_audio

        pred = (F.softmax(logits_audio, dim=1) + F.softmax(logits_image, dim=1)) / 2
        return pred


def imagebind_huge(pretrained=False):
    model = ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
        
    )

    if pretrained:
        weight_path = "data/checkpoint/imagebind_huge.pth"
        if not os.path.exists(weight_path):
            print("Downloading imagebind weights to checkpoint/imagebind_huge.pth ...")
            os.makedirs("data/checkpoint", exist_ok=True)   # <- 关键：确保目录存在
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                weight_path,
                progress=True,
            )

        model.load_state_dict(torch.load(weight_path), strict=False)
    return model
