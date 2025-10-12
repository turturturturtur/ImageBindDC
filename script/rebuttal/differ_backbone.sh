#!/usr/bin/env bash


# --- 基础配置 ---
# 输出和日志目录
OUTPUT_DIR="output/config"
LOG_DIR="logs"

# 如果目录不存在，则创建它们
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# --- 实验参数 ---
# 数据集、ipc、GPU 和 models 列表
DATASETS=("VGG_subset")
IPCS=(10)
CUDA_IDS=(1) # 使用的 GPU 列表
MODELS=(
  "clip_linear.yaml"
  "imagebind_linear.yaml"
)

# --- 新增: 为不同模型类型定义要单独实验的损失函数列表 ---
CLIP_LOSSES=("cf_cos_loss_vision" "avdd_loss" "l2_loss")
IMAGEBIND_LOSSES=("cf_cos_loss" "avdd_loss" "l2_loss")

# 最大并发任务数，设置为 GPU 的数量
MAX_JOBS=${#CUDA_IDS[@]}

# --- 主逻辑 ---
echo "实验执行脚本启动。"
echo "可用 GPU: ${CUDA_IDS[*]}"
echo "最大并发任务数: $MAX_JOBS"
echo "开始时间: $(date)"
echo "-----------------------------------------------------"

# 这是一个独立的函数，负责准备并启动单个实验任务
# 参数: 1:数据集 2:IPC 3:模型文件 4:GPU ID 5:单个损失函数
run_single_task() {
  local dataset="$1"
  local ipc="$2"
  local model_file="$3"
  local gpu_id="$4"
  local loss_function="$5" # 接收单个损失函数作为参数

  # 1. 根据模型名称选择正确的实验模板
  local template_cfg
  if [[ "$model_file" == clip_* ]]; then
    template_cfg="config/experiment/distillation_clip.yaml"
  else
    template_cfg="config/experiment/distillation.yaml"
  fi

  # 2. 准备该任务专用的 YAML 配置文件 (文件名现在包含损失函数名)
  local model_base="${model_file%.yaml}"
  local new_yaml="$OUTPUT_DIR/${dataset}_ipc${ipc}_${model_base}_${loss_function}.yaml"

  # 检查模板文件是否存在
  if [ ! -f "$template_cfg" ]; then
    echo "错误: 模板配置文件 '$template_cfg' 未找到。跳过此任务。" >&2
    return
  fi

  # 拷贝模板并替换参数
  cp "$template_cfg" "$new_yaml"
  sed -i "s/^\s*dataset\s*:.*$/dataset: \"${dataset}\"/" "$new_yaml"
  sed -i "s/^\s*ipc\s*:.*$/ipc: ${ipc}/" "$new_yaml"
  
  # --- 新增: 替换损失函数 ---
  sed -i "s/^\(\s*loss\s*:\).*/\1 \"${loss_function}\"/" "$new_yaml"

  # 3. 准备启动命令 (日志文件名现在也包含损失函数名)
  local model_config_path="config/model/${model_file}"
  local log_file="$LOG_DIR/differ_${dataset}_ipc${ipc}_${model_base}_${loss_function}.log"

  # 检查模型配置文件是否存在
  if [ ! -f "$model_config_path" ]; then
    echo "错误: 模型配置文件 '$model_config_path' 未找到。跳过此任务。" >&2
    return
  fi

  echo "准备启动任务: Dataset=${dataset}, IPC=${ipc}, Model=${model_base}, Loss=${loss_function} on GPU=${gpu_id}"
  echo "  --> 配置文件: ${new_yaml}"
  echo "  --> 日志文件: ${log_file}"

  # 4. 在后台启动 Python 任务
  CUDA_VISIBLE_DEVICES=${gpu_id} \
    python pipeline.py \
      --exp_config "$new_yaml" \
      --model_config "$model_config_path" \
    > "$log_file" 2>&1 &
}

# --- 任务调度器 ---
task_index=0
# 通过四层循环遍历所有实验组合 (新增了 LOSS 循环)
for DATASET in "${DATASETS[@]}"; do
  for IPC in "${IPCS[@]}"; do
    for MODEL_FILE in "${MODELS[@]}"; do
      
      # 根据模型文件选择要遍历的损失函数列表
      current_losses=()
      if [[ "$MODEL_FILE" == "clip_linear.yaml" ]]; then
        current_losses=("${CLIP_LOSSES[@]}")
      elif [[ "$MODEL_FILE" == "imagebind_linear.yaml" ]]; then
        current_losses=("${IMAGEBIND_LOSSES[@]}")
      fi

      # --- 新增: 遍历每一个损失函数，启动一个独立的任务 ---
      for LOSS in "${current_losses[@]}"; do

        # 检查当前后台任务数量是否已达到上限
        while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
          echo "任务队列已满 (正在运行 $MAX_JOBS 个任务)，等待空闲 GPU..."
          wait -n
          sleep 1
        done

        # 使用取模运算为新任务分配一个 GPU
        gpu_index=$((task_index % ${#CUDA_IDS[@]}))
        gpu_id=${CUDA_IDS[$gpu_index]}

        # 调用函数，在后台启动这个新任务 (传入单个 LOSS)
        run_single_task "$DATASET" "$IPC" "$MODEL_FILE" "$gpu_id" "$LOSS"

        # 任务计数器加一
        task_index=$((task_index + 1))
        
        sleep 2
      done
    done
  done
done

# --- 收尾工作 ---
echo "-----------------------------------------------------"
echo "所有任务均已启动，正在等待最后运行的 ${MAX_JOBS} 个或更少的任务完成..."
wait

echo "-----------------------------------------------------"
echo "所有实验任务已全部完成！"
echo "日志文件位于 '$LOG_DIR' 目录。"
echo "结束时间: $(date)"