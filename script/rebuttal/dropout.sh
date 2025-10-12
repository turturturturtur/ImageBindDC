#!/usr/bin/env bash

# 基础 YAML 配置文件（原始模板）
ORIGINAL_YAML="config/experiment/distillation.yaml"
OUTPUT_DIR="output/config"
LOG_DIR="logs"

# 创建必要的文件夹
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 要实验的 dataset、ipc、dropout rates 和 modality
DATASETS=("AVE" "VGG_subset")
IPCS=(1 5 10)
DROPOUT_RATES=(0.01 0.05 0.1 1)
MODALITIES=("audio" "image")

# CUDA 设备 ID 列表（0~7）
CUDA_IDS=(0 1 2 3 4 5 6 7)
num_gpus=${#CUDA_IDS[@]}

echo "Start running dropout experiments with 8 GPUs. Start time: $(date)"

for DATASET in "${DATASETS[@]}"; do
    for IPC in "${IPCS[@]}"; do

        # 生成该组实验专用的 YAML 文件名
        NEW_YAML="$OUTPUT_DIR/${DATASET}_${IPC}.yaml"

        # 拷贝原始 YAML 文件（覆盖同名文件）
        cp "$ORIGINAL_YAML" "$NEW_YAML"

        # 替换 dataset 和 ipc 的值
        sed -i.bak "s/^dataset:.*$/dataset: \"$DATASET\"/" "$NEW_YAML"
        sed -i.bak "s/^ipc:.*$/ipc: $IPC/" "$NEW_YAML"
        rm -f "$NEW_YAML.bak"

        echo "Prepared YAML for dataset=${DATASET}, ipc=${IPC}: $NEW_YAML"

        # 任务计数器，用于 GPU round-robin
        task_idx=0

        # 遍历 dropout rates 和 modalities，一起跑
        for dropout in "${DROPOUT_RATES[@]}"; do
            for modality in "${MODALITIES[@]}"; do

                # 分配 GPU
                gpu_idx=$(( task_idx % num_gpus ))
                GPU="${CUDA_IDS[$gpu_idx]}"

                LOG_FILE="$LOG_DIR/dropout_${DATASET}_ipc${IPC}_rate${dropout}_mod${modality}.log"

                echo "  Launching: dataset=${DATASET}, ipc=${IPC}, dropout=${dropout}, modality=${modality}, GPU=${GPU}"
                CUDA_VISIBLE_DEVICES=${GPU} \
                    python pipeline_dropout.py \
                        --exp_config "$NEW_YAML" \
                        --dropout_rate "${dropout}" \
                        --dropout_modality "${modality}" \
                    > "${LOG_FILE}" 2>&1 &

                task_idx=$((task_idx + 1))
            done
        done

        # 等待本组所有 8 个任务完成，再进入下一个 ipc 或 dataset
        wait
        echo "Completed all tasks for dataset=${DATASET}, ipc=${IPC} at $(date)"
    done
done

echo "All dropout tasks finished. Logs are in $LOG_DIR. End time: $(date)"