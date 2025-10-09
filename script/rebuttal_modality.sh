#!/bin/bash

# --- 环境设置 ---
# 激活 Conda 环境并进入工作目录
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate avdd
cd /home/xmwang/ImagebindDC

# --- 准备工作 ---
# 创建日志目录，如果不存在的话
mkdir -p logs

# 指定要使用的 GPU
GPU_ID=0

# --- 实验 1: AVE 数据集, 移除音频 ---
LOG_FILE="logs/AVE_remove_audio.log"
echo "Running experiment 1/4 on GPU $GPU_ID: dataset=AVE, remove_audio -> $LOG_FILE"
CUDA_VISIBLE_DEVICES=$GPU_ID python main_DM_AV_imagebind_cf_cos_modality_dropout.py \
    --dataset AVE \
    --remove_audio \
    --Iteration 30 \
    --num_eval 3 \
    --interval 1 \
    > "$LOG_FILE" 2>&1

# --- 实验 2: AVE 数据集, 移除图像 ---
LOG_FILE="logs/AVE_remove_image.log"
echo "Running experiment 2/4 on GPU $GPU_ID: dataset=AVE, remove_image -> $LOG_FILE"
CUDA_VISIBLE_DEVICES=$GPU_ID python main_DM_AV_imagebind_cf_cos_modality_dropout.py \
    --dataset AVE \
    --remove_image \
    --Iteration 30 \
    --num_eval 3 \
    --interval 1 \
    > "$LOG_FILE" 2>&1

# --- 实验 3: VGG_subset 数据集, 移除音频 ---
LOG_FILE="logs/VGG_subset_remove_audio.log"
echo "Running experiment 3/4 on GPU $GPU_ID: dataset=VGG_subset, remove_audio -> $LOG_FILE"
CUDA_VISIBLE_DEVICES=$GPU_ID python main_DM_AV_imagebind_cf_cos_modality_dropout.py \
    --dataset VGG_subset \
    --remove_audio \
    --Iteration 30 \
    --num_eval 3 \
    --interval 1 \
    > "$LOG_FILE" 2>&1

# --- 实验 4: VGG_subset 数据集, 移除图像 ---
LOG_FILE="logs/VGG_subset_remove_image.log"
echo "Running experiment 4/4 on GPU $GPU_ID: dataset=VGG_subset, remove_image -> $LOG_FILE"
CUDA_VISIBLE_DEVICES=$GPU_ID python main_DM_AV_imagebind_cf_cos_modality_dropout.py \
    --dataset VGG_subset \
    --remove_image \
    --Iteration 30 \
    --num_eval 3 \
    --interval 1 \
    > "$LOG_FILE" 2>&1

# --- 完成 ---
echo "✅ 所有 4 个实验已顺序完成，日志保存在 logs/ 文件夹中。"