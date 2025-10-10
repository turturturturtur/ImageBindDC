import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
from torch.utils.data import Dataset
from typing import Optional
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
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 epochs: int = 100,
                 val_train_epochs: int = 5,
                 checkpoint_dir: str = 'output/checkpoints'):
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
        self.synthetic_dataset = synthetic_dataset
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.val_train_epochs = val_train_epochs
        
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
        【最终正确版本】
        本方法接收一个被打乱的数据加载器，并在内部重建“按类别”匹配的逻辑。
        """
        self.model.train() # 设置为训练模式
        total_loss = 0.0
        
        # 使用 self.train_loader (它将被设置成 syn_loader)
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Training on Synthetic Data]")
        for batch in progress_bar:
            
            # --- 【核心修改开始】 ---

            # 1. 获取整个批次的数据和标签
            syn_aud_batch = batch['audio'].to(self.device)
            syn_img_batch = batch['frame'].to(self.device)
            labels_batch = batch['label'].to(self.device)

            # 2. 我们将在每个类别上累积梯度，最后一起更新
            self.optimizer.zero_grad()
            
            # 3. 按类别对批次数据进行“解复用” (Demultiplexing)
            #    找出当前批次中出现了哪些类别
            unique_classes_in_batch = torch.unique(labels_batch)
            
            batch_total_loss = 0.0

            # 4. 为每个类别独立计算损失并累积梯度
            for c in unique_classes_in_batch:
                # 4.1. 筛选出当前批次中所有属于类别 c 的合成数据
                class_mask = (labels_batch == c)
                curr_aud_syn = syn_aud_batch[class_mask]
                curr_img_syn = syn_img_batch[class_mask]

                # 4.2. 前向传播 (只对当前类别的数据)
                inputs = {"audio": curr_aud_syn, "image": curr_img_syn}
                features = self.model.forward(inputs, mode="embeddings")
                embed_audio = features["audio"]
                embed_image = features["vision"]
                
                # 4.3. 计算类别 c 的内部对比损失
                loss_c = self.loss_fn(embed_audio, embed_audio, embed_image, embed_image)
                
                # 4.4. 反向传播以【累积】梯度
                #      为了防止样本数多的类别主导梯度，我们将损失按类别数进行平均
                loss_c_avg = loss_c / len(unique_classes_in_batch)
                loss_c_avg.backward()

                batch_total_loss += loss_c.item() # 记录原始损失大小
            
            # 5. 在处理完批次中所有类别的梯度累积后，执行一次优化器步骤
            #    这将使用累积的梯度，同时更新批次中所有被计算过的合成数据
            self.optimizer.step()
            
            # 6. 更新总损失和进度条
            total_loss += batch_total_loss
            progress_bar.set_postfix(batch_loss=batch_total_loss / len(unique_classes_in_batch))
            
            # --- 【核心修改结束】 ---
            
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
                inputs = { "audio": batch['audio'].to(self.device), "image": batch['frame'].to(self.device) }
                labels = batch['label'].to(self.device)

                optimizer_student.zero_grad()
                
                # 直接调用模型的 forward 方法，它会返回最终的预测概率
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