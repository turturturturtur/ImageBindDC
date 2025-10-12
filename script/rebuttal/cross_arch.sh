#!/usr/bin/env bash
set -euo pipefail

# 基础目录 / 文件
OUTPUT_DIR="output/config"
LOG_DIR="logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 数据集、ipc、GPU、models 列表
DATASETS=("AVE" "VGG_subset")
IPCS=(1 5 10)
CUDA_IDS=(0 1 2 3)
num_gpus=${#CUDA_IDS[@]}

# models 列表（按你给的）
MODELS=(
  "clip_convnet.yaml"
  "clip_linear.yaml"
  "clip_mlp.yaml"
  "imagebind_convnet.yaml"
  "imagebind_linear.yaml"
  "imagebind_mlp.yaml"
)

echo "Start running pipeline.py experiments. GPUs: ${CUDA_IDS[*]}. Start time: $(date)"

for DATASET in "${DATASETS[@]}"; do
  for IPC in "${IPCS[@]}"; do

    task_idx=0

    for MODEL_FILE in "${MODELS[@]}"; do

      # 根据 model 前缀选择使用的实验模板
      if [[ "$MODEL_FILE" == clip_* ]]; then
        TEMPLATE_CFG="config/experiment/distillation_clip.yaml"
      else
        TEMPLATE_CFG="config/experiment/distillation.yaml"
      fi

      # 生成唯一的 yaml 名称，便于区分
      MODEL_BASE="${MODEL_FILE%.yaml}"
      NEW_YAML="$OUTPUT_DIR/${DATASET}_ipc${IPC}_${MODEL_BASE}.yaml"

      # 拷贝模板到新 yaml（覆盖）
      cp "$TEMPLATE_CFG" "$NEW_YAML"

      # 尝试替换 dataset: 和 ipc:，如果模板里没有则追加
      if grep -qE '^\s*dataset\s*:' "$NEW_YAML"; then
        sed -i.bak "s/^\s*dataset\s*:.*$/dataset: \"${DATASET}\"/" "$NEW_YAML"
      else
        echo -e "\n# added by run script\ndataset: \"${DATASET}\"" >> "$NEW_YAML"
      fi

      if grep -qE '^\s*ipc\s*:' "$NEW_YAML"; then
        sed -i.bak "s/^\s*ipc\s*:.*$/ipc: ${IPC}/" "$NEW_YAML"
      else
        echo "ipc: ${IPC}" >> "$NEW_YAML"
      fi

      # 清理 sed 备份（mac/linux 兼容处理）
      rm -f "$NEW_YAML.bak" "$NEW_YAML.bak-e" || true

      echo "Prepared YAML: $NEW_YAML  (template: $(basename $TEMPLATE_CFG))"

      # 分配 GPU（轮询）
      gpu_idx=$(( task_idx % num_gpus ))
      GPU="${CUDA_IDS[$gpu_idx]}"

      LOG_FILE="$LOG_DIR/cross_arch_${DATASET}_ipc${IPC}_${MODEL_BASE}.log"

      echo "  Launching: dataset=${DATASET}, ipc=${IPC}, model=${MODEL_FILE}, GPU=${GPU}"
      CUDA_VISIBLE_DEVICES=${GPU} \
        python pipeline.py \
          --exp_config "$NEW_YAML" \
          --model_config "$MODEL_CONFIG_PATH" \
        > "$LOG_FILE" 2>&1 &

      task_idx=$((task_idx + 1))
    done

    # 等待当前 dataset + ipc 下所有后台任务完成（防止跨组抢占 GPU）
    wait
    echo "Completed all models for dataset=${DATASET}, ipc=${IPC} at $(date)"
  done
done

echo "All tasks finished. Logs in: $LOG_DIR. End time: $(date)"