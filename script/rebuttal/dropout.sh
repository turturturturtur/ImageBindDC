#!/usr/bin/env bash

# 基础 YAML 配置文件（原始模板）
ORIGINAL_YAML="config/experiment/distillation.yaml"
OUTPUT_DIR="output/config"
LOG_DIR="logs"

# 创建必要的文件夹
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 要实验的 dataset、ipc、dropout rates 和 modality
DATASETS=("AVE" "VGGsub")
IPCS=(1 5 10)
DROPOUT_RATES=(0.01 0.05 0.1 1)
MODALITIES=("audio" "image")

# CUDA 设备 ID 列表（会循环使用）
CUDA_IDS=(0 1 2 3)

num_gpus=${#CUDA_IDS[@]}

echo "Start running dropout experiments (one task per GPU). Start time: $(date)"

for DATASET in "${DATASETS[@]}"; do
    for IPC in "${IPCS[@]}"; do

        # 生成该组实验专用的 YAML 文件名
        NEW_YAML="$OUTPUT_DIR/${DATASET}_${IPC}.yaml"

        # 拷贝原始 YAML 文件（覆盖同名文件）
        cp "$ORIGINAL_YAML" "$NEW_YAML"

        # 在新的 YAML 文件中替换 dataset 和 ipc 的值
        sed -i.bak "s/^dataset:.*$/dataset: \"$DATASET\"/" "$NEW_YAML"
        sed -i.bak "s/^ipc:.*$/ipc: $IPC/" "$NEW_YAML"
        rm -f "$NEW_YAML.bak"

        echo "Prepared YAML for dataset=${DATASET}, ipc=${IPC}: $NEW_YAML"

        # outer loop 改为 modality：先固定 modality，再并行启动若干（等于 GPU 数量）dropout 任务
        for modality in "${MODALITIES[@]}"; do
            echo "  Running modality=${modality} for dataset=${DATASET}, ipc=${IPC} at $(date)"

            # 启动与 GPU 数相同数量的并行任务（这里 DROPOUT_RATES 长度为 4，num_gpus 也为 4）
            # 确保索引和 GPU 一一对应，保证每个 GPU 只跑 1 个任务
            for idx in "${!DROPOUT_RATES[@]}"; do
                dropout=${DROPOUT_RATES[$idx]}
                gpu_idx=$(( idx % num_gpus ))
                GPU="${CUDA_IDS[$gpu_idx]}"

                LOG_FILE="$LOG_DIR/dropout_${DATASET}_ipc${IPC}_rate${dropout}_mod${modality}.log"

                echo "    Launching dropout=${dropout} -> GPU ${GPU}, log=${LOG_FILE}"
                CUDA_VISIBLE_DEVICES=${GPU} \
                    python pipeline_dropout.py \
                        --exp_config "$NEW_YAML" \
                        --dropout_rate "${dropout}" \
                        --dropout_modality "${modality}" \
                    > "${LOG_FILE}" 2>&1 &
            done

            # 等待本 modality 的所有并行任务完成（保证每个 GPU 在下一 batch 前只有 1 个任务）
            wait
            echo "  Completed modality=${modality} for dataset=${DATASET}, ipc=${IPC} at $(date)"
        done

        echo "Finished dataset=${DATASET}, ipc=${IPC} at $(date)"
    done
done

echo "All dropout tasks finished. Logs are in $LOG_DIR. End time: $(date)"