import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
from torch.utils.data import Dataset
from typing import Optional, Callable
from copy import deepcopy

# 配置一个简单的日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer:
    """
    一个通用的训练器类，封装了标准的训练和验证流程。
    """
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 synthetic_dataset: Dataset,
                 real_dataset: Dataset,
                 real_sampler: Callable,
                 real_batch_size: int,
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 epochs: int = 100,
                 val_train_epochs: int = 5,
                 checkpoint_dir: str = 'output/checkpoints',
                 noise_level: float = 0.0,
                 augment_transform: Optional[Callable] = None):
        """
        初始化 Trainer。

        Args:
            model (nn.Module): 需要训练的模型。
            optimizer (Optimizer): 优化器。
            loss_fn (nn.Module): 损失函数。
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
            lr_scheduler (Scheduler, optional): 学习率调度器。
            device (str): 训练设备 ('cuda' or 'cpu')。
            epochs (int): 训练的总轮数。
            checkpoint_dir (str): 保存模型检查点的目录。
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.val_train_epochs = val_train_epochs
        self.augment_transform = augment_transform
        self.synthetic_dataset = synthetic_dataset
        self.real_dataset = real_dataset
        self.real_sampler = real_sampler
        self.real_batch_size = real_batch_size
        self.noise_level = noise_level
        
        self.best_val_acc = 0.0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        """
        启动完整的训练流程。
        """
        logging.info("Starting training...")
        for epoch in range(1, self.epochs + 1):
            # 训练一个 epoch
            train_loss = self._train_one_epoch(epoch)
            
            # 验证一个 epoch
            val_acc = self._evaluate_synthetic_data_quality(epoch)
            
            # 更新学习率
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # 检查是否是最佳模型并保存
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                logging.info(f"🎉 New best synthetic data found! Validation accuracy: {val_acc:.2f}%")
            
            self._save_checkpoint(epoch, is_best)
        
        logging.info("Training complete.")

    def _train_one_epoch(self, epoch: int) -> float:
        """
        执行单轮训练。
        【最终修正版】
        本方法在 DataLoader 提供的合成数据批次基础上，
        实时采样【形状和类别分布都匹配】的真实数据来计算损失。
        """
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Training on Synthetic Data]")
        for batch in progress_bar:
            
            # 1. 获取合成数据批次及其标签
            syn_aud_batch = batch['audio'].to(self.device)
            syn_img_batch = batch['frame'].to(self.device)
            labels_batch = batch['label'].to(self.device)

            # 2. 【核心修正】: 根据合成批次的类别分布，采样【等量】的真实数据
            
            # 找出当前批次中有哪些类别，以及每个类别有多少样本
            unique_classes, counts = torch.unique(labels_batch, return_counts=True)
            
            list_real_aud_per_class = []
            list_real_img_per_class = []
            
            for c, num_samples in zip(unique_classes, counts):
                # 为类别 c，采样 num_samples 个真实样本
                real_indices = self.real_sampler.sample(c.item(), num_samples.item())
                if not real_indices: continue

                real_data_c = self.real_dataset[real_indices]
                list_real_aud_per_class.append(real_data_c['audio'])
                list_real_img_per_class.append(real_data_c['frame'])

            if not list_real_aud_per_class or not list_real_img_per_class:
                continue
                
            # 将所有采样到的真实数据拼接成一个批次。
            # 注意：此时 aud_real_batch 是按类别排序的
            aud_real_batch_sorted = torch.cat(list_real_aud_per_class, dim=0).to(self.device)
            img_real_batch_sorted = torch.cat(list_real_img_per_class, dim=0).to(self.device)

            # --- 为了让损失函数能公平比较，我们需要将 syn_batch 也按类别排序 ---
            # 这样，排序后的 syn_batch 和 real_batch 就完全对齐了
            sorting_indices = torch.argsort(labels_batch)
            syn_aud_batch_sorted = syn_aud_batch[sorting_indices]
            syn_img_batch_sorted = syn_img_batch[sorting_indices]

            # 调整维度
            if aud_real_batch_sorted.dim()==4:
                aud_real_batch_sorted = aud_real_batch_sorted.unsqueeze(1)


            # REBUTTAL：添加噪声
            if self.noise_level > 0.0:
                noise_aud = torch.randn_like(aud_real_batch_sorted) * self.noise_level
                noise_img = torch.randn_like(img_real_batch_sorted) * self.noise_level
                aud_real_batch_sorted = aud_real_batch_sorted + noise_aud
                img_real_batch_sorted = img_real_batch_sorted + noise_img


            # 3. 清零梯度
            self.optimizer.zero_grad()
            
            # 4. 前向传播 (对排序后、对齐的数据)
            inputs_syn = {"audio": syn_aud_batch_sorted, "image": syn_img_batch_sorted}
            features_syn = self.model.forward(inputs_syn, mode="embeddings")
            embd_aud_syn = features_syn["audio"]
            embd_img_syn = features_syn["vision"]

            inputs_real = {"audio": aud_real_batch_sorted, "image": img_real_batch_sorted}
            features_real = self.model.forward(inputs_real, mode="embeddings")
            embd_aud_real = features_real["audio"].detach()
            embd_img_real = features_real["vision"].detach()
            
            # 5. 计算损失 (现在输入的 Tensor 形状和类别分布都已对齐)
            loss = self.loss_fn(embd_aud_real, embd_aud_syn, embd_img_real, embd_img_syn)
            
            # 6. 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 7. 更新统计
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        logging.info(f"Epoch {epoch} Training Summary: Average Loss: {avg_loss:.4f}")
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, is_best: bool):
        """
        保存检查点。
        【已为你定制，以适应数据蒸馏任务】
        这个方法现在保存的是合成数据和它的优化器状态。
        """
        # 1. 准备要保存的状态字典
        state = {
            'epoch': epoch,
            'best_val_acc': self.best_val_acc,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'synthetic_audio_data': self.synthetic_dataset.audio,
            'synthetic_image_data': self.synthetic_dataset.images,
            'synthetic_labels': self.synthetic_dataset.labels # 如果标签也是可学习的
        }
        
        if self.lr_scheduler:
            state['scheduler_state_dict'] = self.lr_scheduler.state_dict()
            
        # 2. 定义文件名
        filename = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        
        # 3. 保存检查点
        torch.save(state, filename)
        logging.info(f"Saved synthetic data checkpoint: {filename}")
        
        # 4. 如果是当前最好的结果，额外保存一份为 'best_syn_data.pth'
        if is_best:
            best_filename = os.path.join(self.checkpoint_dir, "best_syn_data.pth")
            torch.save(state, best_filename)
            logging.info(f"🎉 Saved best synthetic data to: {best_filename}")

    def _evaluate_synthetic_data_quality(self, epoch: int) -> float:
        """
        通过只训练模型分类头并在真实测试集上评估，来衡量合成数据的质量。
        【已为你模型的 ImageBindClassifier 结构精确定制】
        """
        logging.info(f"--- Starting validation for epoch {epoch}: Evaluating synthetic data quality ---")

        # --- 步骤 1: 准备一个用于评估的“学生”模型副本 ---
        # 我们使用原始教师模型的一个深拷贝，以确保原始模型不受影响
        student_model = deepcopy(self.model)
        student_model.to(self.device)

        # --- 步骤 2: 冻结特征提取器，只训练分类头 ---
        # 冻结所有参数
        for param in student_model.parameters():
            param.requires_grad = False
        
        # 然后，只解冻你指定的分类头的参数
        logging.info("Unfreezing classifier weights for validation training...")
        for param in student_model.classifier_audio.parameters():
            param.requires_grad = True
        for param in student_model.classifier_image.parameters():
            param.requires_grad = True
        
        # 创建一个只包含可训练参数的优化器
        trainable_params = filter(lambda p: p.requires_grad, student_model.parameters())
        optimizer_student = torch.optim.Adam(trainable_params, lr=0.001)
        
        loss_fn_student = nn.CrossEntropyLoss()
        # 训练数据加载器就是 self.train_loader (syn_loader)
        inner_train_loader = self.train_loader 

        # --- 步骤 3: 内部快速训练循环 ---
        logging.info(f"Training classifier head for {self.val_train_epochs} epochs on synthetic data...")
        student_model.train() # 将学生模型设为训练模式
        for inner_epoch in range(self.val_train_epochs):
            for batch in inner_train_loader:
                # 从 batch 中获取干净的合成数据
                syn_aud_batch = batch['audio'].to(self.device)
                syn_img_batch = batch['frame'].to(self.device)
                labels = batch['label'].to(self.device)

                # 在送入学生模型前，进行实时数据增强
                if self.augment_transform:
                    syn_img_batch = self.augment_transform(syn_img_batch)
                    # 你也可以为音频做增强
                
                inputs = { "audio": syn_aud_batch, "image": syn_img_batch }
                
                optimizer_student.zero_grad()
                
                # 使用增强后的数据进行训练
                predictions = student_model.forward(inputs)
                
                loss = loss_fn_student(predictions, labels)
                
                loss.backward()
                optimizer_student.step()
        
        # --- 步骤 4: 在真实测试集上评估训练好的学生模型 ---
        logging.info("Evaluating the trained student model on the real test set...")
        student_model.eval() # 将学生模型设为评估模式
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_loader: # self.val_loader 是真实的 test_loader
                inputs = { "audio": batch['audio'].to(self.device), "image": batch['frame'].to(self.device) }
                labels = batch['label'].to(self.device)
                
                # 获得最终的预测概率
                outputs = student_model.forward(inputs)
                # 从概率中得到预测的类别
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        logging.info(f"Validation Summary: Student model accuracy on real test data: {accuracy:.2f}%")
        
        del student_model # 释放副本占用的显存
        return accuracy