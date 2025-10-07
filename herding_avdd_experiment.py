#!/usr/bin/env python3
"""
Herding算法增强的AVDD实验
解决小样本蒸馏中的代表性问题
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

class HerdingSelector:
    """Herding算法选择器"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def herding_selection(self, features, num_select):
        """
        Herding算法选择代表性样本
        Args:
            features: 特征矩阵 [N, D]
            num_select: 要选择的样本数量
        Returns:
            selected_indices: 选择的样本索引
        """
        features = features.detach().cpu().numpy()
        n_samples = features.shape[0]
        
        if num_select >= n_samples:
            return torch.arange(n_samples)
        
        # 计算特征均值
        mean_feature = np.mean(features, axis=0, keepdims=True)
        
        # 初始化
        selected_indices = []
        remaining_indices = list(range(n_samples))
        
        # 第一个选择最接近均值的样本
        similarities = cosine_similarity(mean_feature, features)[0]
        first_idx = np.argmax(similarities)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # 迭代选择
        selected_features = features[selected_indices]
        
        for _ in range(1, num_select):
            if not remaining_indices:
                break
                
            # 计算当前选择集的特征均值
            current_mean = np.mean(selected_features, axis=0, keepdims=True)
            
            # 计算剩余样本与当前均值的相似度
            remaining_features = features[remaining_indices]
            similarities = cosine_similarity(current_mean, remaining_features)[0]
            
            # 选择最相似的样本
            best_idx_in_remaining = np.argmax(similarities)
            best_idx = remaining_indices[best_idx_in_remaining]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            selected_features = features[selected_indices]
        
        return torch.tensor(selected_indices, dtype=torch.long)

class HerdingAVDD:
    """Herding增强的AVDD算法"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.herding = HerdingSelector(device)
    
    def run_herding_experiment(self, num_real_samples=100, num_syn_samples=20, 
                             feature_dim=1024, iterations=200):
        """运行Herding增强的AVDD实验"""
        
        print("=" * 80)
        print("🎯 Herding增强AVDD实验")
        print("=" * 80)
        print(f"📊 真实样本: {num_real_samples}")
        print(f"🎯 合成样本: {num_syn_samples}")
        print(f"⚙️  特征维度: {feature_dim}")
        print(f"🖥️  设备: {device}")
        print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 生成真实数据
        print("\n📊 生成真实数据...")
        real_audio_features = F.normalize(torch.randn(num_real_samples, feature_dim, device=device), dim=-1)
        real_text_features = F.normalize(torch.randn(num_real_samples, feature_dim, device=device), dim=-1)
        
        # 创建真实相关性
        correlation_strength = 0.8
        for i in range(num_real_samples):
            real_text_features[i] = correlation_strength * real_audio_features[i] + \
                                   (1 - correlation_strength) * torch.randn(feature_dim, device=device)
        real_text_features = F.normalize(real_text_features, dim=-1)
        
        # 使用Herding选择代表性样本
        print("\n🎯 使用Herding选择代表性样本...")
        audio_indices = self.herding.herding_selection(real_audio_features, num_syn_samples)
        text_indices = self.herding.herding_selection(real_text_features, num_syn_samples)
        
        print(f"  选择音频样本索引: {audio_indices[:10].tolist()}...")
        print(f"  选择文本样本索引: {text_indices[:10].tolist()}...")
        
        selected_audio = real_audio_features[audio_indices]
        selected_text = real_text_features[text_indices]
        
        # 初始化合成数据（基于选择的样本）
        print("\n🎯 初始化合成数据...")
        syn_audio = selected_audio.clone() + 0.1 * torch.randn_like(selected_audio)
        syn_text = selected_text.clone() + 0.1 * torch.randn_like(selected_text)
        
        syn_audio.requires_grad = True
        syn_text.requires_grad = True
        
        optimizer = torch.optim.Adam([syn_audio, syn_text], lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
        
        loss_history = []
        
        # 运行蒸馏
        print("\n🔬 开始Herding增强蒸馏...")
        start_time = time.time()
        
        for iteration in range(iterations):
            # 生成合成特征
            syn_audio_feat = F.normalize(syn_audio, dim=-1)
            syn_text_feat = F.normalize(syn_text, dim=-1)
            
            # 计算AVDD损失
            # 1. 分布匹配损失
            target_audio = real_audio_features[audio_indices]
            target_text = real_text_features[text_indices]
            loss_dm = F.mse_loss(syn_audio_feat, target_audio) + \
                     F.mse_loss(syn_text_feat, target_text)
            
            # 2. 模态间一致性损失
            real_sim = torch.matmul(target_audio, target_text.t())
            syn_sim = torch.matmul(syn_audio_feat, syn_text_feat.t())
            loss_icm = F.mse_loss(syn_sim, real_sim)
            
            # 3. 跨模态全局匹配损失
            real_gap = target_audio.mean(0) - target_text.mean(0)
            syn_gap = syn_audio_feat.mean(0) - syn_text_feat.mean(0)
            loss_cgm = F.mse_loss(syn_gap, real_gap)
            
            # 4. Herding正则化（保持多样性）
            # 防止所有合成样本变得相同
            audio_diversity = torch.std(syn_audio_feat, dim=0).mean()
            text_diversity = torch.std(syn_text_feat, dim=0).mean()
            loss_diversity = -0.1 * (audio_diversity + text_diversity)  # 负号表示鼓励多样性
            
            loss = loss_dm + 5.0 * loss_icm + 2.0 * loss_cgm + loss_diversity
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_history.append(loss.item())
            
            if iteration % 40 == 0 or iteration == iterations - 1:
                print(f"迭代 {iteration:3d}/{iterations}, 损失: {loss.item():.6f}")
        
        elapsed_time = time.time() - start_time
        
        # 评估最终性能
        print("\n📈 评估最终检索性能...")
        final_audio_feat = F.normalize(syn_audio, dim=-1)
        final_text_feat = F.normalize(syn_text, dim=-1)
        
        # 在完整数据集上评估
        results = evaluate_herding_retrieval(final_audio_feat, final_text_feat, 
                                           real_audio_features, real_text_features)
        
        # 打印详细结果
        print("\n" + "=" * 80)
        print("📊 Herding增强实验结果")
        print("=" * 80)
        
        print(f"\n🔍 检索性能:")
        print("-" * 50)
        for k in [1, 5, 10]:
            print(f"  Recall@{k}: A2T={results[f'R@{k}_a2t']:.4f} ({results[f'R@{k}_a2t']*100:.2f}%), "
                  f"T2A={results[f'R@{k}_t2a']:.4f} ({results[f'R@{k}_t2a']*100:.2f}%)")
        
        print(f"\n📊 训练统计:")
        print(f"  初始损失: {loss_history[0]:.6f}")
        print(f"  最终损失: {loss_history[-1]:.6f}")
        print(f"  损失下降: {(loss_history[0] - loss_history[-1]):.6f}")
        print(f"  训练时间: {elapsed_time:.2f}秒")
        
        # 保存结果
        results_dict = {
            'method': 'herding_avdd',
            'num_real_samples': num_real_samples,
            'num_syn_samples': num_syn_samples,
            'feature_dim': feature_dim,
            'iterations': iterations,
            'retrieval_results': results,
            'loss_history': loss_history,
            'elapsed_time': elapsed_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        filename = f'herding_avdd_{feature_dim}d_{num_syn_samples}s_results.pth'
        torch.save(results_dict, filename)
        
        return results_dict

def evaluate_herding_retrieval(syn_audio, syn_text, real_audio, real_text, k_values=[1, 5, 10]):
    """评估Herding增强的检索性能"""
    # 计算相似度矩阵
    sim_matrix = torch.matmul(real_audio, syn_text.t())
    sim_t2a = torch.matmul(real_text, syn_audio.t())
    
    results = {}
    
    # Audio-to-Text检索 (真实音频 vs 合成文本)
    _, top_indices_a2t = torch.topk(sim_matrix, k=min(max(k_values), sim_matrix.size(1)), dim=1)
    
    # Text-to-Audio检索 (真实文本 vs 合成音频)
    _, top_indices_t2a = torch.topk(sim_t2a, k=min(max(k_values), sim_t2a.size(1)), dim=1)
    
    for k in k_values:
        # 实际k值不能超过合成样本数量
        actual_k = min(k, syn_audio.size(0))
        
        # A2T召回率
        correct_a2t = 0
        for i in range(min(len(real_audio), len(syn_text))):
            if i < top_indices_a2t.size(1):
                # 找到最近的合成样本
                closest_syn_idx = top_indices_a2t[i, 0]
                if closest_syn_idx < len(syn_text):
                    # 计算真实样本间的相似度
                    true_sim = torch.matmul(real_audio[i:i+1], real_text[i:i+1].t())
                    pred_sim = torch.matmul(real_audio[i:i+1], syn_text[closest_syn_idx:closest_syn_idx+1].t())
                    if abs(true_sim.item() - pred_sim.item()) < 0.1:
                        correct_a2t += 1
        recall_a2t = correct_a2t / min(len(real_audio), len(syn_text))
        
        # T2A召回率
        correct_t2a = 0
        for i in range(min(len(real_text), len(syn_audio))):
            if i < top_indices_t2a.size(1):
                closest_syn_idx = top_indices_t2a[i, 0]
                if closest_syn_idx < len(syn_audio):
                    true_sim = torch.matmul(real_text[i:i+1], real_audio[i:i+1].t())
                    pred_sim = torch.matmul(real_text[i:i+1], syn_audio[closest_syn_idx:closest_syn_idx+1].t())
                    if abs(true_sim.item() - pred_sim.item()) < 0.1:
                        correct_t2a += 1
        recall_t2a = correct_t2a / min(len(real_text), len(syn_audio))
        
        results[f'R@{k}_a2t'] = recall_a2t
        results[f'R@{k}_t2a'] = recall_t2a
    
    return results

if __name__ == "__main__":
    print("🏆 Herding增强AVDD实验")
    print("=" * 80)
    
    # 运行Herding实验
    herding_engine = HerdingAVDD(device)
    
    # ConvNet实验
    results_convnet = herding_engine.run_herding_experiment(
        num_real_samples=100, num_syn_samples=20, feature_dim=6272, iterations=200
    )
    
    print("\n" + "=" * 80)
    
    # ImageBind实验  
    results_imagebind = herding_engine.run_herding_experiment(
        num_real_samples=100, num_syn_samples=20, feature_dim=1024, iterations=200
    )
    
    print("\n" + "=" * 80)
    print("🎯 Herding实验总结")
    print("=" * 80)
    
    print(f"{'特征类型':<12} {'R@1_A2T':<10} {'R@1_T2A':<10} {'R@5_A2T':<10} {'R@5_T2A':<10}")
    print("-" * 60)
    
    for results in [results_convnet, results_imagebind]:
        ft = 'convnet' if results['feature_dim'] == 6272 else 'imagebind'
        r1_a2t = results['retrieval_results']['R@1_a2t']
        r1_t2a = results['retrieval_results']['R@1_t2a']
        r5_a2t = results['retrieval_results']['R@5_a2t']
        r5_t2a = results['retrieval_results']['R@5_t2a']
        
        print(f"{ft:<12} {r1_a2t:<10.4f} {r1_t2a:<10.4f} {r5_a2t:<10.4f} {r5_t2a:<10.4f}")