#!/bin/bash

# 输出和日志文件夹
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 定义要实验的 dataset 和 ipc
DATASETS=("VGG_subset" "AVE")
IPCS=(1 5 10)

# 定义 CUDA
CUDA_IDS=(0 1 2 3)

# 遍历 dataset 和 ipc
for DATASET in "${DATASETS[@]}"; do
    for IPC in "${IPCS[@]}"; do

        # 并行运行 pipeline.py
        for i in ${!CUDA_IDS[@]}; do
            CUDA_VISIBLE_DEVICES=${CUDA_IDS[$i]} \
            python pipeline.py \
                --dataset "$DATASET" \
                --ipc $IPC \
                > "$LOG_DIR/dsdmloss_${DATASET}_${IPC}.log" 2>&1 &
        done

        # 等待本组任务完成再继续下一组
        wait
    done
done

echo "All tasks finished. Logs are in $LOG_DIR"