#!/bin/bash

# 原始 YAML 文件
ORIGINAL_YAML="config/experiment/distillation.yaml"
OUTPUT_DIR="output/config"
LOG_DIR="logs"

# 创建必要文件夹
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 定义要实验的 dataset 和 ipc
DATASETS=("VGG_subset" "AVE")
IPCS=(5 10)

# 定义 CUDA 和 noise_level
CUDA_IDS=(0 1 2 3)
NOISE_LEVELS=(0.01 0.1 0.2 0.5)

# 遍历 dataset 和 ipc
for DATASET in "${DATASETS[@]}"; do
    for IPC in "${IPCS[@]}"; do

        # 生成 YAML 文件名
        NEW_YAML="$OUTPUT_DIR/${DATASET}_${IPC}.yaml"

        # 拷贝原始 YAML
        cp "$ORIGINAL_YAML" "$NEW_YAML"

        # 替换 dataset 和 ipc
        sed -i.bak "s/^dataset:.*$/dataset: \"$DATASET\"/" "$NEW_YAML"
        sed -i.bak "s/^ipc:.*$/ipc: $IPC/" "$NEW_YAML"
        rm "$NEW_YAML.bak"

        # 并行运行 pipeline_noise.py
        for i in ${!CUDA_IDS[@]}; do
            CUDA_VISIBLE_DEVICES=${CUDA_IDS[$i]} \
            python pipeline_noise.py \
                --exp_config "$NEW_YAML" \
                --noise_level ${NOISE_LEVELS[$i]} \
                > "$LOG_DIR/noise_${DATASET}_${IPC}_${NOISE_LEVELS[$i]}.log" 2>&1 &
        done

        # 等待本组任务完成再继续下一组
        wait
    done
done

echo "All tasks finished. Logs are in $LOG_DIR"