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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 添加这个缺失的类
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
    def __init__(self, audio_dir: str, captions_df: pd.DataFrame, vocab: Dict, sr=22050, n_mels=64):
        self.audio_dir = audio_dir
        self.captions_df = captions_df
        self.vocab = vocab
        self.sr = sr
        self.n_mels = n_mels

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        row = self.captions_df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['file_name'])
        y, _ = librosa.load(audio_path, sr=self.sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        audio_tensor = torch.tensor(librosa.power_to_db(mel_spec), dtype=torch.float32).unsqueeze(0)

        caption = row['caption_1']
        words = caption.lower().split()
        word_ids = [self.vocab.get(word, self.vocab['<unk>']) for word in words]
        text_tensor = torch.tensor(word_ids, dtype=torch.long)

        return {'audio': audio_tensor, 'text': text_tensor}

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
    audios = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    audios_stacked = torch.stack(audios)
    return {'audio': audios_stacked, 'text': texts_padded}

def get_high_performance_dataloader(audio_dir: str, captions_file: str, batch_size: int, num_samples: int) -> Tuple[DataLoader, Dict]:
    df = pd.read_csv(captions_file)
    if num_samples:
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    vocab = build_vocab(df['caption_1'].tolist())
    dataset = RawClothoDataset(audio_dir, df, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    return dataloader, vocab

def precompute_real_features(dataloader: DataLoader, feature_extractor, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算整个真实数据集的特征。
    
    Args:
        dataloader: 加载真实原始数据（频谱图、文本ID）的DataLoader。
        feature_extractor: 特征提取模型 (例如 ImageBindExtractor 或 ConvNetGRU)。
        device: 'cuda' 或 'cpu'。

    Returns:
        包含所有音频和文本特征的两个大型张量。
    """
    feature_extractor.to(device)
    feature_extractor.eval()
    
    all_audio_features = []
    all_text_features = []
    
    print(f"🚀 开始预计算真实特征，使用模型: {feature_extractor.__class__.__name__}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="正在提取特征"):
            audio_raw = batch['audio'].to(device)
            text_raw = batch['text'].to(device)
            
            # 根据模型类型调用不同的特征提取方法
            if isinstance(feature_extractor, ImageBindExtractor):
                # ImageBindExtractor 需要特殊处理
                # 注意：这里的 extract_*_features 需要适配为接收批次数据
                audio_feats = feature_extractor.extract_audio_features(audio_raw)
                text_feats = feature_extractor.extract_text_features(text_raw)
            else: # 适用于 ConvNetGRU 等双塔模型
                audio_feats = feature_extractor.forward_audio(audio_raw)
                text_feats = feature_extractor.forward_text(text_raw)

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
        # 1. 将输入的词汇ID序列通过Embedding层转换为特征向量序列
        embedded_text = self.text_embedding(text)
        
        # 2. 将特征向量序列输入GRU
        # GRU的第二个返回值h_n是最后一个时间步的隐藏状态，形状为(num_layers, batch_size, hidden_size)
        _, h_n = self.text_gru(embedded_text)
        
        # 3. h_n.squeeze(0)将形状变为(batch_size, hidden_size)，这通常被用作整个序列的特征表示
        text_feat = h_n.squeeze(0)
        
        # 4. 归一化特征
        return F.normalize(text_feat, dim=-1)

# 替换旧的 ImageBindExtractor 类
class ImageBindExtractor:
    """ImageBind特征提取器 (模拟)"""
    def __init__(self, feature_dim=1024, pretrained=False, device='cpu'): # 增加参数以匹配调用
        self.feature_dim = feature_dim
        self.device = device
        if pretrained:
            print("模拟加载预训练的 ImageBind 模型...")

    def extract_audio_features(self, audio_data: torch.Tensor) -> torch.Tensor:
        batch_size = audio_data.size(0)
        features = torch.randn(batch_size, self.feature_dim, device=self.device)
        return F.normalize(features, dim=-1)

    def extract_text_features(self, text_data: torch.Tensor) -> torch.Tensor:
        batch_size = text_data.size(0)
        features = torch.randn(batch_size, self.feature_dim, device=self.device)
        return F.normalize(features, dim=-1)

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
    
    def distill(self, real_audio_features, real_text_features, 
                method='avdd', feature_space='imagebind', 
                num_syn_samples=20, iterations=200):
        """标准蒸馏流程"""
        
        print(f"\n🔬 开始{method} on {feature_space}蒸馏...")
        
        # 使用Herding选择核心集
        audio_indices, audio_coreset = self.herding.select_coreset(real_audio_features, num_syn_samples)
        text_indices, text_coreset = self.herding.select_coreset(real_text_features, num_syn_samples)
        
        # 基于核心集初始化合成数据
        syn_audio = audio_coreset.clone().detach().requires_grad_(True)
        syn_text = text_coreset.clone().detach().requires_grad_(True)
        
        # 选择损失函数
        if method == 'avdd':
            loss_fn = AVDDLoss()
        else:  # imagebind_dc
            loss_fn = ImageBindDCLoss(device=self.device)
        
        optimizer = torch.optim.Adam([syn_audio, syn_text], lr=0.01)
        
        loss_history = []
        
        # ### MODIFICATION START ###
        # 移除了无效的 feature_extractor 创建和传递逻辑，因为损失函数是在固定的特征空间上计算的。
        for iteration in range(iterations):
            loss = loss_fn(
                (real_audio_features, real_text_features),
                (syn_audio, syn_text)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            
            if iteration % 50 == 0:
                print(f"迭代 {iteration}/{iterations}, 损失: {loss:.6f}")
        # ### MODIFICATION END ###
        
        return {
            'synthetic_audio': syn_audio.detach(),
            'synthetic_text': syn_text.detach(),
            'loss_history': loss_history
        }

# ===================== 检索模型训练 =====================

class RetrievalModel(nn.Module):
    """检索模型"""
    
    def __init__(self, input_dim, feature_dim=512):
        super().__init__()
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
    
    def forward(self, audio, text):
        audio_feat = F.normalize(self.audio_encoder(audio), dim=-1)
        text_feat = F.normalize(self.text_encoder(text), dim=-1)
        return audio_feat, text_feat

class RetrievalTrainer:
    """检索模型训练器"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def train(self, synthetic_dataset, feature_dim=512, epochs=50):
        """训练检索模型"""
        
        print("\n🎯 训练检索模型...")
        
        dataloader = DataLoader(synthetic_dataset, batch_size=8, shuffle=True)
        
        input_dim = synthetic_dataset[0]['audio'].shape[0]
        model = RetrievalModel(input_dim, feature_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        def info_nce_loss(audio_feat, text_feat, temperature=0.07):
            logits = torch.matmul(audio_feat, text_feat.t()) / temperature
            labels = torch.arange(len(audio_feat), device=self.device)
            
            loss_a2t = F.cross_entropy(logits, labels)
            loss_t2a = F.cross_entropy(logits.t(), labels)
            
            return (loss_a2t + loss_t2a) / 2
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                audio = batch['audio'].to(self.device)
                text = batch['text'].to(self.device)
                
                audio_feat, text_feat = model(audio, text)
                loss = info_nce_loss(audio_feat, text_feat)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"训练轮次 {epoch}/{epochs}, 平均损失: {total_loss/len(dataloader):.4f}")
        
        return model

# ===================== 标准评估 =====================

class StandardEvaluator:
    """标准评估器"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def evaluate(self, model, real_audio_features, real_text_features):
        """标准Recall@K评估"""
        
        print("\n🎯 标准检索评估...")
        
        model.eval()
        with torch.no_grad():
            audio_feat, text_feat = model(real_audio_features, real_text_features)
            
            # 计算相似度矩阵
            sim_matrix = torch.matmul(audio_feat, text_feat.t())
            
            results = {}
            
            # Audio-to-Text检索
            _, top_indices_a2t = torch.topk(sim_matrix, k=10, dim=1)
            
            # Text-to-Audio检索
            _, top_indices_t2a = torch.topk(sim_matrix.t(), k=10, dim=1)
            
            for k in [1, 5, 10]:
                # A2T召回率
                correct_a2t = 0
                for i in range(len(real_audio_features)):
                    if i in top_indices_a2t[i, :k]:
                        correct_a2t += 1
                recall_a2t = correct_a2t / len(real_audio_features)
                
                # T2A召回率
                correct_t2a = 0
                for i in range(len(real_text_features)):
                    if i in top_indices_t2a[i, :k]:
                        correct_t2a += 1
                recall_t2a = correct_t2a / len(real_text_features)
                
                results[f'R@{k}_a2t'] = recall_a2t
                results[f'R@{k}_t2a'] = recall_t2a
        
        return results

# ===================== 主实验流程 =====================

class ClothoExperiment:
    """Clotho数据集完整实验"""
    
    def __init__(self, audio_dir: str, captions_file: str, device='cuda'):
        self.device = device
        self.audio_dir = audio_dir
        self.captions_file = captions_file
        self.vocab_size = 0 # 将在运行时更新
        self.distiller = DataDistillation(device)
        self.trainer = RetrievalTrainer(device)
        self.evaluator = StandardEvaluator(device)
    
    def run_complete_experiment(self, num_real_samples=2893, num_syn_samples=20):
        """执行完整四阶段实验"""
        
        print("🏆 Clotho数据集标准数据蒸馏实验")
        print("=" * 80)
        
        # ===================== 阶段零：准备真实数据和特征 =====================
        print("📊 阶段零：加载真实数据并预计算特征...")

        # 1. 创建加载真实原始数据的 DataLoader
        # 注意：这里我们加载 num_real_samples 个样本用于蒸馏
        real_dataloader, vocab = get_high_performance_dataloader(
            audio_dir=self.audio_dir,
            captions_file=self.captions_file,
            batch_size=32, # 可以根据显存调整
            num_samples=num_real_samples
        )
        self.vocab_size = len(vocab) # 更新词汇表大小

        # 2. 准备两种不同的特征提取器
        imagebind_extractor = ImageBindExtractor(pretrained=True, device=self.device)
        # 使用正确的构造函数创建 ConvNetGRU
        convnet_extractor = ConvNetGRU(vocab_size=self.vocab_size, feature_dim=512).to(self.device)

        # 3. 执行特征预计算，得到四个真实的特征库
        print("\n--- 计算ImageBind特征空间 ---")
        real_imagebind_audio, real_imagebind_text = precompute_real_features(
            real_dataloader, imagebind_extractor, self.device
        )
        
        print("\n--- 计算ConvNet特征空间 ---")
        real_convnet_audio, real_convnet_text = precompute_real_features(
            real_dataloader, convnet_extractor, self.device
        )
        
        print(f"\nImageBind 特征维度: Audio-{real_imagebind_audio.shape}, Text-{real_imagebind_text.shape}")
        print(f"ConvNet 特征维度: Audio-{real_convnet_audio.shape}, Text-{real_convnet_text.shape}")

        # ===================== 阶段一：四种蒸馏方法 =====================
        print("\n🔬 阶段一：执行四种数据蒸馏方法...")
        
        # 方法一：AVDD on ConvNet
        result1 = self.distiller.distill(
            real_convnet_audio, real_convnet_text,
            method='avdd', feature_space='convnet',
            num_syn_samples=num_syn_samples, iterations=1000
        )
        
        # 方法二：ImageBindDC on ImageBind
        result2 = self.distiller.distill(
            real_imagebind_audio, real_imagebind_text,
            method='imagebind_dc', feature_space='imagebind',
            num_syn_samples=num_syn_samples, iterations=1000
        )
        
        # 方法三：AVDD on ImageBind
        result3 = self.distiller.distill(
            real_imagebind_audio, real_imagebind_text,
            method='avdd', feature_space='imagebind',
            num_syn_samples=num_syn_samples, iterations=1000
        )
        
        # 方法四：ImageBindDC on ConvNet
        result4 = self.distiller.distill(
            real_convnet_audio, real_convnet_text,
            method='imagebind_dc', feature_space='convnet',
            num_syn_samples=num_syn_samples, iterations=1000
        )
        
        # ===================== 阶段二 & 三：训练并评估检索模型 =====================
        print("\n🔬 阶段二 & 三：在合成数据上训练并评估检索模型...")
        
        distilled_datasets = [
            ("AVDD_ConvNet", result1, (real_convnet_audio, real_convnet_text)),
            ("ImageBindDC_ImageBind", result2, (real_imagebind_audio, real_imagebind_text)),
            ("AVDD_ImageBind", result3, (real_imagebind_audio, real_imagebind_text)),
            ("ImageBindDC_ConvNet", result4, (real_convnet_audio, real_convnet_text))
        ]
        
        final_results = {}
        
        for name, synthetic_data, real_eval_features in distilled_datasets:
            print(f"\n--- 正在处理方法: {name} ---")
            
            # 准备合成数据集
            syn_dataset = SyntheticDataset(
                synthetic_data['synthetic_audio'], 
                synthetic_data['synthetic_text']
            )

            # 训练学生模型
            # 注意：学生模型输入维度应与合成特征维度匹配
            input_dim = synthetic_data['synthetic_audio'].shape[1]
            feature_dim = 512 # 假设我们统一训练一个512维的学生模型
            trained_model = self.trainer.train(syn_dataset, input_dim=input_dim, feature_dim=feature_dim, epochs=100)
            
            # 在真实的评估数据上进行评估
            # 注意：评估时，学生模型需要处理与它训练时相同特征空间的数据
            eval_audio_features, eval_text_features = real_eval_features
            
            # 为了演示，我们使用一部分真实特征作为评估集
            eval_subset_size = 500
            eval_results = self.evaluator.evaluate(
                trained_model, 
                eval_audio_features[:eval_subset_size], 
                eval_text_features[:eval_subset_size]
            )
            final_results[name] = eval_results
        
        # 阶段四：结果汇总
        print("\n" + "=" * 80)
        print("📊 最终2x2对比矩阵")
        print("=" * 80)
        
        print(f"{'方法':<25} {'Recall@1_A2T':<12} {'Recall@1_T2A':<12} {'Recall@5_A2T':<12} {'Recall@5_T2A':<12}")
        print("-" * 80)
        
        for name, result in results.items():
            print(f"{name:<25} {result['R@1_a2t']:<12.4f} {result['R@1_t2a']:<12.4f} "
                  f"{result['R@5_a2t']:<12.4f} {result['R@5_t2a']:<12.4f}")
        
        # 保存完整结果
        final_report = {
            'num_real_samples': num_real_samples,
            'num_syn_samples': num_syn_samples,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        torch.save(final_report, f'clotho_experiment_{num_syn_samples}s_results.pth')
        print(f"\n💾 完整结果已保存: clotho_experiment_{num_syn_samples}s_results.pth")
        
        return results

# ===================== 主入口 =====================

if __name__ == "__main__":
    print("🚀 Clotho数据集标准数据蒸馏实验启动")

    # --- 用户需要提供的路径 ---
    AUDIO_DIR = './clotho_data/development/'
    CAPTIONS_FILE = './clotho_data/clotho_captions_development.csv'
    # -------------------------

    if not os.path.exists(AUDIO_DIR) or not os.path.exists(CAPTIONS_FILE):
         print(f"错误：找不到数据路径 {AUDIO_DIR} 或 {CAPTIONS_FILE}")
    else:
         # 创建实验实例
         experiment = ClothoExperiment(
             audio_dir=AUDIO_DIR,
             captions_file=CAPTIONS_FILE,
             device=device
         )
         # 运行实验
         experiment.run_complete_experiment(num_real_samples=100, num_syn_samples=20)