#!/bin/bash

# 基础 YAML 配置文件
ORIGINAL_YAML="config/experiment/distillation.yaml"
OUTPUT_DIR="output/config"
LOG_DIR="logs"

# 创建必要的文件夹
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 定义要实验的 dataset 和 ipc
# 注意：IPC 增加了 1
DATASETS=("VGG_subset" "AVE")
IPCS=(1 5 10)

# 定义 CUDA 设备 ID 和 mismatch rates
# 注意：这里从 NOISE_LEVELS 换成了 MISMATCH_RATES
CUDA_IDS=(0 1 2 3)
MISMATCH_RATES=(0.01 0.05 0.1 0.2)

# 遍历 dataset 和 ipc
for DATASET in "${DATASETS[@]}"; do
    for IPC in "${IPCS[@]}"; do

        # 生成该组实验专用的 YAML 文件名
        NEW_YAML="$OUTPUT_DIR/${DATASET}_${IPC}.yaml"

        # 拷贝原始 YAML 文件
        cp "$ORIGINAL_YAML" "$NEW_YAML"

        # 在新的 YAML 文件中替换 dataset 和 ipc 的值
        sed -i.bak "s/^dataset:.*$/dataset: \"$DATASET\"/" "$NEW_YAML"
        sed -i.bak "s/^ipc:.*$/ipc: $IPC/" "$NEW_YAML"
        rm "$NEW_YAML.bak"

        # 并行运行 mismatch 实验
        # 每个 mismatch_rate 在一个指定的 GPU 上运行
        for i in ${!CUDA_IDS[@]}; do
            CUDA_VISIBLE_DEVICES=${CUDA_IDS[$i]} \
            python pipeline_mismatch.py \
                --exp_config "$NEW_YAML" \
                --mismatch_rate ${MISMATCH_RATES[$i]} \
                > "$LOG_DIR/mismatch_${DATASET}_${IPC}_${MISMATCH_RATES[$i]}.log" 2>&1 &
        done

        # 等待本组 (4个) 并行任务全部完成，再开始下一组
        wait
    done
done

echo "All mismatch tasks finished. Logs are in $LOG_DIR"