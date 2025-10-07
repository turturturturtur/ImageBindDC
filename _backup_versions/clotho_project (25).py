#!/usr/bin/env python3
"""
Clotho数据集标准数据蒸馏实验
遵循技术规格书的完整四阶段实现
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, List
import argparse
import wandb
from utils.train_utils_DM import get_network_imagebind
from nets.imagebind.models.imagebind_model import ModalityType

import sys
# 假设您的项目根目录是 'audio-visual-mater'
PROJECT_ROOT = '/autodl-tmp/audio-visual-mater/'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 从您的本地文件夹导入必要的模块
from utils.train_utils_DM import get_network_imagebind
from nets.imagebind.models.imagebind_model import ModalityType
from nets.imagebind import data # 【关键】从您的nets文件夹导入data



class SyntheticDataset(Dataset):
    """用于已蒸馏的合成特征的数据集"""
    def __init__(self, synthetic_audio: torch.Tensor, synthetic_text: torch.Tensor):
        self.audio = synthetic_audio
        self.text = synthetic_text

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return {
            'audio': self.audio[idx],
            'text': self.text[idx],
            'label': idx
        }

# 添加这个缺失的函数 (它依赖于一个修改版的RawClothoDataset，这里一并提供)
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm import tqdm

class RawClothoDataset(Dataset):
    def __init__(self,
                 audio_dir: str,
                 captions_df: pd.DataFrame,
                 vocab: Dict,
                 sr: int = 22050,
                 n_mels: int = 64,
                 max_time_steps: int = 1024):
        self.audio_dir = audio_dir
        self.captions_df = captions_df
        self.vocab = vocab
        self.sr = sr
        self.n_mels = n_mels
        self.max_time_steps = max_time_steps

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        row = self.captions_df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['file_name'])

        # 1. 加载音频并计算梅尔频谱图
        y, _ = librosa.load(audio_path, sr=self.sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec)          # shape: (n_mels, T)

        # 2. 统一时间维度：padding 或截断
        if mel_spec_db.shape[1] < self.max_time_steps:
            pad_width = self.max_time_steps - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :self.max_time_steps]

        audio_mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)  # shape: (1, n_mels, T_fixed)

        # 3. 文本序列转 ID
        caption_string = row['caption_1']
        words = caption_string.lower().split()
        word_ids = [self.vocab.get(word, self.vocab['<unk>']) for word in words]
        text_ids_tensor = torch.tensor(word_ids, dtype=torch.long)

        return {
            'audio_mel': audio_mel_tensor,
            'text_ids': text_ids_tensor,
            'audio_path': audio_path,
            'raw_caption': caption_string
        }
def build_vocab(captions: List[str], min_freq=5) -> Dict:
    word_counts = {}
    for caption in captions:
        for word in caption.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = {word: i+2 for i, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    return vocab

def collate_fn(batch):
    # 为 ConvNetGRU 处理数据
    audio_mels = [item['audio_mel'] for item in batch]
    text_ids = [item['text_ids'] for item in batch]
    texts_padded = pad_sequence(text_ids, batch_first=True, padding_value=0)
    audios_stacked = torch.stack(audio_mels)

    # 为 ImageBind 收集原始数据 (保持为List)
    audio_paths = [item['audio_path'] for item in batch]
    raw_captions = [item['raw_caption'] for item in batch]
    
    return {
        'audio_mel': audios_stacked,
        'text_ids': texts_padded,
        'audio_path': audio_paths,
        'raw_caption': raw_captions
    }
    
def get_high_performance_dataloader(audio_dir: str, captions_file: str, batch_size: int, num_samples: int, vocab: Dict = None) -> Tuple[DataLoader, Dict]:
    """
    创建一个高性能的Dataloader。
    新增功能：可以接收一个已有的vocab，如果vocab为None，则会自己构建。
    """
    df = pd.read_csv(captions_file)
    if num_samples:
        # 注意：对于评估集，我们通常加载全部数据，因此num_samples应为None
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    if vocab is None:
        print("未提供词汇表，正在从字幕中构建...")
        vocab = build_vocab(df['caption_1'].tolist())
    else:
        print("使用已提供的词汇表。")

    dataset = RawClothoDataset(audio_dir, df, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    # 始终返回使用的dataloader和vocab
    return dataloader, vocab

def precompute_real_features(dataloader: DataLoader, feature_extractor, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(feature_extractor, nn.Module):
        feature_extractor.to(device)
        feature_extractor.eval()
    
    all_audio_features, all_text_features = [], []
    print(f"🚀 开始预计算真实特征，使用模型: {feature_extractor.__class__.__name__}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="正在提取特征"):
            if isinstance(feature_extractor, ImageBindExtractor):
                audio_feats = feature_extractor.extract_audio_features(batch['audio_path'])
                text_feats = feature_extractor.extract_text_features(batch['raw_caption'])
            else: 
                audio_data = batch['audio_mel'].to(device)
                text_data = batch['text_ids'].to(device)
                audio_feats = feature_extractor.forward_audio(audio_data)
                text_feats = feature_extractor.forward_text(text_data)

            all_audio_features.append(audio_feats.cpu())
            all_text_features.append(text_feats.cpu())
            
    print("✅ 真实特征预计算完成！")
    return torch.cat(all_audio_features, dim=0).to(device), torch.cat(all_text_features, dim=0).to(device)
# ===================== 特征提取器 =====================

class ConvNetGRU(nn.Module):
    """ConvNet+GRU双塔模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, feature_dim: int = 512):
        super().__init__()
        
        # 音频塔 (这部分保持不变)
        self.audio_tower = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim)
        )
        
        # 文本塔 (重构为独立的层，而不是一个Sequential)
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.text_gru = nn.GRU(embedding_dim, feature_dim, batch_first=True)
    
    def forward(self, x):
        """简化forward函数用于特征提取"""
        return x  # 直接返回输入，因为这里只是特征转换
    
    def forward_audio(self, audio):
        return F.normalize(self.audio_tower(audio), dim=-1)
    
    def forward_text(self, text: torch.Tensor) -> torch.Tensor:
        text = torch.clamp(text, max=self.text_embedding.num_embeddings - 1)
        # 1. 将输入的词汇ID序列通过Embedding层转换为特征向量序列
        embedded_text = self.text_embedding(text)
        
        # 2. 将特征向量序列输入GRU
        # GRU的第二个返回值h_n是最后一个时间步的隐藏状态，形状为(num_layers, batch_size, hidden_size)
        _, h_n = self.text_gru(embedded_text)
        
        # 3. h_n.squeeze(0)将形状变为(batch_size, hidden_size)，这通常被用作整个序列的特征表示
        text_feat = h_n.squeeze(0)
        
        # 4. 归一化特征
        return F.normalize(text_feat, dim=-1)

class ImageBindExtractor:
    """真实的 ImageBind 特征提取器 (模仿您的本地项目)"""
    def __init__(self, pretrained=False, device='cpu'):
        self.device = device
        self.model = None
        self.embed_func = None

        if pretrained:
            print("正在使用您本地的 get_network_imagebind 函数加载模型...")
            
            class MockArgs:
                def __init__(self):
                    self.arch_frame = 'imagebind'
                    self.arch_classifier = 'ensemble'
                    self.cls_num = 10
                    self.weights_classifier = ''
                    self.input_modality = 'av'

            mock_args = MockArgs()
            
            nets, _ = get_network_imagebind(mock_args)
            net_imagebind, _ = nets
            
            self.model = net_imagebind
            self.model.to(self.device)
            self.model.eval()

            self.embed_func = self.model.module.embed if torch.cuda.device_count() > 1 else self.model.embed
            print("✅ 真实 ImageBind 模型加载完成！")

    def extract_audio_features(self, audio_path_list: List[str]) -> torch.Tensor:
        if self.model is None: raise RuntimeError("ImageBind模型未加载！")
        # 【关键】调用您本地的 data.py 中的函数
        inputs = { ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path_list, self.device) }
        with torch.no_grad():
            embeddings = self.embed_func(inputs)
        return embeddings[ModalityType.AUDIO].detach()

    def extract_text_features(self, raw_caption_list: List[str]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("ImageBind模型未加载！")
    
        print("raw_caption_list 前 3 条：", raw_caption_list[:3])
    
        # 1️⃣ 拿到 token id
        text_tokens = data.load_and_transform_text(raw_caption_list, self.device)
    
        # 2️⃣ 打印最大 id，看是否真的 ≥ 49408
        max_id = text_tokens.max().item()
        print("max token id =", max_id, " 越界？", max_id >= 49408)
    
        # 3️⃣ 兜底 clamp
        text_tokens = torch.clamp(text_tokens, max=49407)
    
        inputs = {ModalityType.TEXT: text_tokens}
        with torch.no_grad():
            embeddings = self.embed_func(inputs)
        return embeddings[ModalityType.TEXT].detach()

# ===================== 标准Herding算法 =====================

class StandardHerdingSelector:
    """标准Herding算法的规范实现"""

    def __init__(self, device='cuda'):
        self.device = device

    def select_coreset(self, features, coreset_size):
        if coreset_size >= features.shape[0]:
            return torch.arange(features.shape[0]), features

        # 1. 计算一次全体特征的均值
        overall_mean = features.mean(dim=0)

        selected_indices = []
        available_indices = list(range(features.shape[0]))

        # 2. 初始化当前核心集的均值
        current_coreset_mean = torch.zeros_like(overall_mean)

        for i in range(coreset_size):
            # 3. 计算当前均值与目标均值的“误差向量”
            error_vector = overall_mean - current_coreset_mean

            # 4. 在剩余样本中，寻找与误差向量内积最大的那个
            available_features = features[available_indices]
            projections = torch.matmul(available_features, error_vector)

            best_idx_in_remaining = torch.argmax(projections)
            selected_idx = available_indices.pop(best_idx_in_remaining)

            selected_indices.append(selected_idx)

            # 5. 更新当前核心集的均值
            current_coreset_mean = (current_coreset_mean * i + features[selected_idx]) / (i + 1)

        selected_indices = torch.tensor(selected_indices)
        return selected_indices, features[selected_indices]

# ===================== 损失函数 =====================

class AVDDLoss:
    """AVDD损失函数"""
    
    def __init__(self, lam_icm=10.0, lam_cgm=10.0):
        self.lam_icm = lam_icm
        self.lam_cgm = lam_cgm
    
    def __call__(self, real_features, synthetic_features):
        """计算AVDD损失"""
        real_audio, real_text = real_features
        syn_audio, syn_text = synthetic_features
        
        # 1. 分布匹配损失 (DM Loss)
        loss_dm = F.mse_loss(syn_audio.mean(0), real_audio.mean(0)) + \
                 F.mse_loss(syn_text.mean(0), real_text.mean(0))
        
        # 2. 模态间一致性损失 (ICM Loss) - 修正后匹配统计量
        # 假设输入的特征已经被归一化
        real_sim_matrix = torch.matmul(real_audio, real_text.t())
        syn_sim_matrix = torch.matmul(syn_audio, syn_text.t())
        
        loss_icm = F.mse_loss(syn_sim_matrix.mean(), real_sim_matrix.mean()) + \
                   F.mse_loss(syn_sim_matrix.std(), real_sim_matrix.std())
        
        # 3. 跨模态全局匹配损失 (CGM Loss)
        real_gap = real_audio.mean(0) - real_text.mean(0)
        syn_gap = syn_audio.mean(0) - syn_text.mean(0)
        loss_cgm = F.mse_loss(syn_gap, real_gap)
        
        return loss_dm + self.lam_icm * loss_icm + self.lam_cgm * loss_cgm
        
class ImageBindDCLoss:
    """
    ImageBindDC损失函数 - 修正后采用真实的CFD实现。
    """
    def __init__(self, lam_cross=1.0, lam_joint=1.0, num_freqs=4096, device='cuda'):
        self.lam_cross = lam_cross
        self.lam_joint = lam_joint
        self.num_freqs = num_freqs
        self.device = device
        self._t_cache = {} # 用于缓存不同维度的随机频率向量t

    def _get_t(self, feature_dim):
        """根据特征维度获取或生成随机频率向量t"""
        if feature_dim not in self._t_cache:
            # 从标准正态分布中采样随机频率
            self._t_cache[feature_dim] = torch.randn(feature_dim, self.num_freqs, device=self.device)
        return self._t_cache[feature_dim]

    def compute_cfd(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        计算两个特征分布之间的特征函数距离 (CFD)。
        公式: L2^2(Φ_x(t), Φ_y(t))
        """
        feature_dim = feat1.shape[1]
        t = self._get_t(feature_dim) # 获取随机频率向量

        # 1. 将特征投影到随机频率上 (t^T * x)
        proj1 = torch.matmul(feat1, t)
        proj2 = torch.matmul(feat2, t)

        # 2. 计算特征函数 (通过采样近似期望 E[exp(j * proj)])
        # exp(j*z) = cos(z) + j*sin(z)
        # E[cos(z)] 和 E[sin(z)] 通过在batch维度上求均值来近似
        phi1_real = torch.cos(proj1).mean(dim=0)
        phi1_imag = torch.sin(proj1).mean(dim=0)
        phi2_real = torch.cos(proj2).mean(dim=0)
        phi2_imag = torch.sin(proj2).mean(dim=0)

        # 3. 计算两个复数特征函数之间的L2距离的平方
        dist_sq = (phi1_real - phi2_real)**2 + (phi1_imag - phi2_imag)**2
        
        # 4. 在所有频率上求和，得到最终的CFD损失
        return dist_sq.sum()

    def __call__(self, real_features, synthetic_features):
        real_audio, real_text = real_features
        syn_audio, syn_text = synthetic_features

        # 由于真实数据量远大于合成数据，我们需要从中采样一个子集来进行匹配
        num_syn = len(syn_audio)
        rand_idx = torch.randperm(len(real_audio))[:num_syn]
        real_audio_subset = real_audio[rand_idx]
        real_text_subset = real_text[rand_idx]

        # 1. 单模态分布匹配 (Uni-modal Distribution Matching)
        loss_uni = self.compute_cfd(real_audio_subset, syn_audio) + \
                   self.compute_cfd(real_text_subset, syn_text)
        
        # 2. 跨模态分布匹配 (Cross-modal Distribution Matching)
        # 遵循论文公式 CFD(Real_Audio + Syn_Text, Real_Text + Syn_Audio)
        mix_dist1 = real_audio_subset + syn_text
        mix_dist2 = real_text_subset + syn_audio
        loss_cross = self.compute_cfd(mix_dist1, mix_dist2)
        
        # 3. 联合分布匹配 (Joint Distribution Matching)
        # 拼接特征，匹配其联合分布
        joint_real = torch.cat([real_audio_subset, real_text_subset], dim=1)
        joint_syn = torch.cat([syn_audio, syn_text], dim=1)
        loss_joint = self.compute_cfd(joint_real, joint_syn)
        
        return loss_uni + self.lam_cross * loss_cross + self.lam_joint * loss_joint
# ===================== 数据蒸馏 =====================

class DataDistillation:
    """数据蒸馏引擎"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.herding = StandardHerdingSelector(device)
    
    # 在 DataDistillation 类中
    # 在 DataDistillation 类中
    def distill(self, real_audio_features, real_text_features, args, method='avdd', feature_space='imagebind'):
        """标准蒸馏流程，签名已修改为接收args"""
        
        print(f"\n🔬 开始{method} on {feature_space}蒸馏...")
        
        num_syn_samples = args.ipc
        iterations = args.distill_iter
    
        audio_indices, audio_coreset = self.herding.select_coreset(real_audio_features, num_syn_samples)
        text_indices, text_coreset = self.herding.select_coreset(real_text_features, num_syn_samples)
        
        syn_audio = audio_coreset.clone().detach().requires_grad_(True)
        syn_text = text_coreset.clone().detach().requires_grad_(True)
        
        # 【最终修正】确保ImageBindDCLoss也使用args中的权重
        if method == 'avdd':
            loss_fn = AVDDLoss(lam_icm=args.lam_icm, lam_cgm=args.lam_cgm)
        else:  # imagebind_dc
            loss_fn = ImageBindDCLoss(lam_cross=args.lam_cross, lam_joint=args.lam_joint, device=self.device)
        
        optimizer = torch.optim.Adam([syn_audio, syn_text], lr=args.distill_lr)
        
        loss_history = []
        
        for iteration in range(iterations):
            loss = loss_fn(
                (real_audio_features, real_text_features),
                (syn_audio, syn_text)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            
            if iteration % 50 == 0 or iteration == iterations - 1:
                if not args.disable_wandb:
                    wandb.log({f"distill_loss/{method}_{feature_space}": loss.item()})
                print(f"迭代 {iteration}/{iterations}, 损失: {loss.item():.6f}")
        
        return {
            'synthetic_audio': syn_audio.detach(),
            'synthetic_text': syn_text.detach(),
            'loss_history': loss_history
        }
# ===================== 检索模型训练 =====================
class ImageBindAsRetriever(nn.Module):
    """冻结 ImageBind + 强化版轻量映射头，用于合成特征训练"""

    def __init__(self, feature_dim=512, device='cuda'):
        super().__init__()
        self.device = device

        # 映射头（输入维度固定为 1024），进一步强化
        self.audio_head = nn.Sequential(
            nn.Linear(1024, 2048),  # 增加中间层宽度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),  # 新增一个隐藏层，并保持宽度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),  # 再一个隐藏层
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, feature_dim) # 输出到目标特征维度
        )
        self.text_head = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, feature_dim)
        )

    def forward(self, audio, text):
        audio = audio.to(self.device)
        text  = text.to(self.device)
        audio_feat = F.normalize(self.audio_head(audio), dim=-1)
        text_feat = F.normalize(self.text_head(text), dim=-1)
        return audio_feat, text_feat

class RetrievalModel(nn.Module):
    """检索模型 - 强化版"""
    
    def __init__(self, input_dim, feature_dim=512):
        super().__init__()
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),  # 输入层，宽度增大
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),   # 新增隐藏层，并增加宽度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),   # 再一个隐藏层
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, feature_dim) # 输出到目标特征维度
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, feature_dim)
        )
    def forward(self, audio, text):
        audio_feat = F.normalize(self.audio_encoder(audio), dim=-1)
        text_feat = F.normalize(self.text_encoder(text), dim=-1)
        return audio_feat, text_feat

class RetrievalTrainer:
    """检索模型训练器"""

    def __init__(self, device='cuda'):
        self.device = device

    def info_nce_loss(self, audio_feat, text_feat, temperature=0.07):
        logits = torch.matmul(audio_feat, text_feat.t()) / temperature
        labels = torch.arange(len(audio_feat), device=self.device)
        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.t(), labels)
        return (loss_a2t + loss_t2a) / 2

    # 这里是修改的部分
    def train(self, syn_dataset, input_dim, args, feature_dim=512, model_type='ImageBindAsRetriever'): # <-- 新增 model_type 参数
        print(f"\n🎯 训练学生模型... (epochs: {args.student_epochs}, lr: {args.student_lr}, Model Type: {model_type})") # <-- 打印模型类型
        dataloader = DataLoader(syn_dataset, batch_size=8, shuffle=True)

        if model_type == 'ImageBindAsRetriever':
            model = ImageBindAsRetriever(feature_dim=feature_dim, device=self.device).to(self.device)
        elif model_type == 'RetrievalModel':
            model = RetrievalModel(input_dim=input_dim, feature_dim=feature_dim).to(self.device)
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.student_lr)

        model.train()
        for epoch in range(args.student_epochs):
            total_loss = 0
            for batch in dataloader:
                audio, text = batch['audio'].to(self.device), batch['text'].to(self.device)
                audio_feat, text_feat = model(audio, text)

                loss = self.info_nce_loss(audio_feat, text_feat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch == args.student_epochs - 1:
                print(f"训练轮次 {epoch}/{args.student_epochs}, 平均损失: {total_loss/len(dataloader):.4f}")

        return model

# ===================== 标准评估 =====================

class StandardEvaluator:
    """标准评估器"""

    def __init__(self, device='cuda'):
        self.device = device

    # 这里是修改的部分
    def evaluate(self,
                 model,
                 real_audio_features: torch.Tensor, # <-- 明确类型提示为张量
                 real_text_features: torch.Tensor): # <-- 明确类型提示为张量
        """
        标准 Recall@K 评估。
        现在统一接收预计算的特征张量。
        """

        print("\n🎯 标准检索评估...")

        model.eval()
        with torch.no_grad():
            # ---------- 统一获取特征 ----------
            if real_audio_features is None or real_text_features is None:
                raise ValueError("评估模型时必须提供 real_audio_features 和 real_text_features")
            
            # 统一调用模型的 forward 方法，传入特征张量
            audio_feat, text_feat = model(real_audio_features, real_text_features)

            audio_feat = audio_feat.to(self.device)
            text_feat  = text_feat.to(self.device)

            # ---------- 计算相似度 ----------
            sim_matrix = torch.matmul(audio_feat, text_feat.t())

            results = {}
            for k in [1, 5, 10, 20]:
                # A2T
                _, topk_a2t = torch.topk(sim_matrix, k=k, dim=1)
                recall_a2t = torch.mean(
                    (torch.arange(len(audio_feat), device=self.device).unsqueeze(1) == topk_a2t).any(dim=1).float()
                ).item()

                # T2A
                _, topk_t2a = torch.topk(sim_matrix.t(), k=k, dim=1)
                recall_t2a = torch.mean(
                    (torch.arange(len(text_feat), device=self.device).unsqueeze(1) == topk_t2a).any(dim=1).float()
                ).item()

                results[f'R@{k}_a2t'] = recall_a2t
                results[f'R@{k}_t2a'] = recall_t2a

        return results
# ===================== 主实验流程 =====================

class ClothoExperiment:
    """Clotho数据集完整实验"""
    
    def __init__(self, dev_audio_dir: str, dev_captions_file: str, eval_audio_dir: str, eval_captions_file: str, device='cuda'):
            self.device = device
            # 开发集路径
            self.dev_audio_dir = dev_audio_dir
            self.dev_captions_file = dev_captions_file
            # 评估集路径
            self.eval_audio_dir = eval_audio_dir
            self.eval_captions_file = eval_captions_file
            
            self.vocab = None # 用于存储词汇表
            self.distiller = DataDistillation(device)
            self.trainer = RetrievalTrainer(device)
            self.evaluator = StandardEvaluator(device)
    
    # 在 ClothoExperiment 类中
    def run_complete_experiment(self, args):
        """
        执行完整四阶段实验。
        函数签名已修改为接收argparse解析出的参数对象。
        """
        print("🏆 Clotho数据集标准数据蒸馏实验")
        print("=" * 80)
        
        # ===================== 阶段零：准备数据和特征提取器 =====================
        print("📊 阶段零：加载数据并准备特征提取器...")
    
        # 1. 加载开发集数据，并【构建】词汇表
        print("\n--- 加载开发集 (用于蒸馏) ---")
        dev_dataloader, self.vocab = get_high_performance_dataloader(
            audio_dir=self.dev_audio_dir,
            captions_file=self.dev_captions_file,
            batch_size=32,
            num_samples=args.dev_samples
        )
        
        # 2. 加载评估集数据，并【复用】开发集的词汇表
        print("\n--- 加载评估集 (用于最终测试) ---")
        eval_dataloader, _ = get_high_performance_dataloader(
            audio_dir=self.eval_audio_dir,
            captions_file=self.eval_captions_file,
            batch_size=32,
            num_samples=None, # 加载全部评估集样本
            vocab=self.vocab  # 关键：传入开发集构建的词汇表
        )
    
        # 3. 准备两种不同的特征提取器
        imagebind_extractor = ImageBindExtractor(pretrained=True, device=self.device)
        convnet_extractor = ConvNetGRU(vocab_size=len(self.vocab), feature_dim=512).to(self.device)
    
        # ===================== 阶段零(续)：特征预计算 =====================
        print("\n--- 在开发集上预计算【蒸馏用】特征 ---")
        real_imagebind_audio, real_imagebind_text = precompute_real_features(dev_dataloader, imagebind_extractor, self.device)
        real_convnet_audio, real_convnet_text = precompute_real_features(dev_dataloader, convnet_extractor, self.device)
        
        print("\n--- 在评估集上预计算【评估用】特征 ---")
        eval_imagebind_audio, eval_imagebind_text = precompute_real_features(eval_dataloader, imagebind_extractor, self.device)
        eval_convnet_audio, eval_convnet_text = precompute_real_features(eval_dataloader, convnet_extractor, self.device)
        
        # ===================== 阶段一：四种蒸馏方法 (在开发集特征上进行) =====================
        print("\n🔬 阶段一：执行四种数据蒸馏方法...")
        
        # 【修正】将args对象传递给distill方法，以使用其中定义的学习率和损失权重
        result1 = self.distiller.distill(real_convnet_audio, real_convnet_text, args, method='avdd', feature_space='convnet')
        result2 = self.distiller.distill(real_imagebind_audio, real_imagebind_text, args, method='imagebind_dc', feature_space='imagebind')
        result3 = self.distiller.distill(real_imagebind_audio, real_imagebind_text, args, method='avdd', feature_space='imagebind')
        result4 = self.distiller.distill(real_convnet_audio, real_convnet_text, args, method='imagebind_dc', feature_space='convnet')
        
        # ===================== 阶段二 & 三：训练并【在评估集上】评估检索模型 =====================
        print("\n🔬 阶段二 & 三：在合成数据上训练并使用【评估集】评估...")
        
        distilled_datasets = [
            ("AVDD_ConvNet", result1, (eval_convnet_audio, eval_convnet_text), 'convnet'), # <-- 新增一个元素来表示特征空间
            ("ImageBindDC_ImageBind", result2, (eval_imagebind_audio, eval_imagebind_text), 'imagebind'), # <-- 新增
            ("AVDD_ImageBind", result3, (eval_imagebind_audio, eval_imagebind_text), 'imagebind'), # <-- 新增
            ("ImageBindDC_ConvNet", result4, (eval_convnet_audio, eval_convnet_text), 'convnet') # <-- 新增
        ]
        
        final_results = {}
        # 循环签名也要更新，以接收新的 feature_space 参数
        for name, synthetic_data, real_eval_features, current_feature_space in distilled_datasets: # <-- 循环签名更新
            print(f"\n--- 正在处理方法: {name} ---")
            syn_dataset = SyntheticDataset(synthetic_data['synthetic_audio'], synthetic_data['synthetic_text'])
            
            input_dim = synthetic_data['synthetic_audio'].shape[1]
            
            # 根据当前的特征空间选择学生模型类型
            if current_feature_space == 'imagebind': # <--- 修改后的判断逻辑
                student_model_type = 'ImageBindAsRetriever'
            elif current_feature_space == 'convnet': # <--- 修改后的判断逻辑
                student_model_type = 'RetrievalModel'
            else:
                raise ValueError(f"未知特征空间: {current_feature_space}")

            # 【修正】将args对象传递给train方法，以使用其中定义的学习率和epochs
            # 同时也传递选择的学生模型类型
            trained_model = self.trainer.train(
                syn_dataset, 
                input_dim, # input_dim 会是 512 (ConvNet) 或 1024 (ImageBind)
                args, 
                feature_dim=512, # 学生模型的输出特征维度，这通常是固定的
                model_type=student_model_type 
            )
            
            # 关键：使用从未见过的评估集特征进行评估
            eval_audio_features, eval_text_features = real_eval_features
            
            # 由于 StandardEvaluator.evaluate 已经统一接口为接收特征张量，
            # 这里直接调用即可，不再需要 isinstane 判断
            eval_results = self.evaluator.evaluate(trained_model, eval_audio_features, eval_text_features)
            
            final_results[name] = eval_results
        
        # ===================== 阶段四：结果汇总 =====================
        print("📊 最终对比矩阵（Recall@1/5/10/20）")
        print("=" * 110)
        
        print(f"{'方法':<25} "
              f"{'R@1_A2T':<8} {'R@1_T2A':<8} "
              f"{'R@5_A2T':<8} {'R@5_T2A':<8} "
              f"{'R@10_A2T':<9} {'R@10_T2A':<9} "
              f"{'R@20_A2T':<9} {'R@20_T2A':<9}")
        print("-" * 110)
        
        for name, result in final_results.items():
            print(f"{name:<25} "
                  f"{result['R@1_a2t']:<8.4f} {result['R@1_t2a']:<8.4f} "
                  f"{result['R@5_a2t']:<8.4f} {result['R@5_t2a']:<8.4f} "
                  f"{result['R@10_a2t']:<9.4f} {result['R@10_t2a']:<9.4f} "
                  f"{result['R@20_a2t']:<9.4f} {result['R@20_t2a']:<9.4f}")
        # 保存完整结果
        final_report = {
            'args': vars(args),
            'results': final_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        filename = f'clotho_experiment_ipc{args.ipc}_seed{args.seed}_results.pth'
        torch.save(final_report, filename)
        print(f"\n💾 完整结果已保存: {filename}")
        
        return final_results
# ===================== 主入口 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clotho 数据集标准数据蒸馏实验")

    # --- 路径与数据参数 ---
    parser.add_argument('--base_path', type=str, default='/root/autodl-tmp/audio-visual-master/data/Clotho/', help='Clotho数据集的根目录')
    parser.add_argument('--dev_audio_dir_name', type=str, default='development', help='开发集音频文件夹名')
    parser.add_argument('--eval_audio_dir_name', type=str, default='evaluation', help='评估集音频文件夹名')
    parser.add_argument('--dev_captions_name', type=str, default='clotho_captions_development.csv', help='开发集字幕文件名')
    parser.add_argument('--eval_captions_name', type=str, default='clotho_captions_evaluation.csv', help='评估集字幕文件名')

    # --- 核心实验参数 ---
    parser.add_argument('--ipc', type=int, default=20, help='每个类别蒸馏出的合成样本数量 (num_syn_samples)')
    parser.add_argument('--dev_samples', type=int, default=2893, help='用于蒸馏的开发集样本数 (num_real_samples)')
    parser.add_argument('--distill_iter', type=int, default=5000, help='蒸馏优化的迭代次数')
    parser.add_argument('--student_epochs', type=int, default=100, help='训练学生检索模型的轮数')
    parser.add_argument('--distill_lr', type=float, default=0.001, help='蒸馏阶段优化器的学习率')
    parser.add_argument('--student_lr', type=float, default=0.001, help='学生模型训练阶段的学习率')

    # --- AVDD 损失权重 ---
    parser.add_argument('--lam_icm', type=float, default=10.0, help='AVDD损失中, 模态间一致性损失(ICM)的权重')
    parser.add_argument('--lam_cgm', type=float, default=10.0, help='AVDD损失中, 跨模态全局匹配损失(CGM)的权重')
    
    # --- ImageBindDC/CFD 损失权重 ---
    parser.add_argument('--lam_cross', type=float, default=1.0, help='CFD损失中, 跨模态分布匹配损失的权重')
    parser.add_argument('--lam_joint', type=float, default=1.0, help='CFD损失中, 联合分布匹配损失的权重')

    # --- 可复现性与监控 ---
    parser.add_argument('--seed', type=int, default=42, help='全局随机种子')
    parser.add_argument('--num_runs', type=int, default=1, help='使用不同随机种子重复实验的次数')
    parser.add_argument('--wandb_project', type=str, default="clotho_distillation_final", help='Wandb项目名称')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb的实体/用户名 (可选)')
    parser.add_argument('--disable_wandb', action='store_true', help='如果设置, 则禁用wandb日志')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()

    # 主循环，用于多次随机种子实验
    for i in range(args.num_runs):
        current_seed = args.seed + i
        print(f"\n{'='*30}  运行第 {i+1}/{args.num_runs} 次, 随机种子: {current_seed}  {'='*30}\n")
        
        # 设置随机种子
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)

        # 拼接完整的路径
        DEV_AUDIO_DIR = os.path.join(args.base_path, args.dev_audio_dir_name)
        DEV_CAPTIONS_FILE = os.path.join(args.base_path, args.dev_captions_name)
        EVAL_AUDIO_DIR = os.path.join(args.base_path, args.eval_audio_dir_name)
        EVAL_CAPTIONS_FILE = os.path.join(args.base_path, args.eval_captions_name)

        # 检查路径
        if not all(os.path.exists(p) for p in [DEV_AUDIO_DIR, DEV_CAPTIONS_FILE]):
            print(f"错误：请确保开发集数据路径都存在！")
            print(f"  - 开发集音频: {DEV_AUDIO_DIR}")
            print(f"  - 开发集字幕: {DEV_CAPTIONS_FILE}")
            # 评估集路径的检查可以放到实验流程内部，因为可能只做蒸馏不做评估
            continue # 如果开发集不存在，则跳过本次运行

        # 初始化Wandb
        if not args.disable_wandb:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=args,
                name=f"run_{i+1}_seed_{current_seed}",
                reinit=True # 允许多次在同一个脚本中初始化
            )
    
        # 创建并运行实验
        experiment = ClothoExperiment(
            dev_audio_dir=DEV_AUDIO_DIR,
            dev_captions_file=DEV_CAPTIONS_FILE,
            eval_audio_dir=EVAL_AUDIO_DIR,
            eval_captions_file=EVAL_CAPTIONS_FILE,
            device=device
        )
        experiment.run_complete_experiment(args) # 将所有参数传递给实验主函数

        if not args.disable_wandb:
            wandb.finish()