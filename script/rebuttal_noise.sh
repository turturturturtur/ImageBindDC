source  ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate avdd
cd /home/xmwang/ImagebindDC

# ❗️ 对于AVE来讲，image模态的embedding在-0.4~+0.5，noise_embedding=0.1 等于加入N（0，0.1^2）的噪声
# ❗️ 对于VGG10k来讲，image模态的embedding在-0.28~+0.43，noise_embedding=0.1 等于加入N（0，0.1^2）的噪声

# 创建日志目录
mkdir -p logs

# GPU 设备列表
gpus=(1 2 3)
gpu_idx=0

# # 固定 ipc=1, 改变 noise_embedding
# for noise in 0.01 0.5 0.2; do
#     gpu=${gpus[$gpu_idx]}
#     log_file="logs/AVE_ipc1_noise${noise}.log"
#     echo "Running on GPU $gpu with ipc=1, noise_embedding=$noise -> $log_file"
#     CUDA_VISIBLE_DEVICES=$gpu python main_DM_AV_imagebind_cf_cos_random_noise.py \
#         --dataset AVE \
#         --Iteration 30 \
#         --noise_embedding $noise \
#         --num_eval 3 \
#         --interval 1 \
#         --ipc 1 \
#         > "$log_file" 2>&1 &
#     gpu_idx=$(( (gpu_idx + 1) % ${#gpus[@]} ))
# done

# wait

# # 固定 ipc=1, 改变 noise_embedding
# for noise in 0.01 0.5 0.2; do
#     gpu=${gpus[$gpu_idx]}
#     log_file="logs/VGG_ipc1_noise${noise}.log"
#     echo "Running on GPU $gpu with ipc=1, noise_embedding=$noise -> $log_file"
#     CUDA_VISIBLE_DEVICES=$gpu python main_DM_AV_imagebind_cf_cos_random_noise.py \
#         --dataset VGG_subset \
#         --Iteration 30 \
#         --noise_embedding $noise \
#         --num_eval 3 \
#         --interval 1 \
#         --ipc 1 \
#         > "$log_file" 2>&1 &
#     gpu_idx=$(( (gpu_idx + 1) % ${#gpus[@]} ))
# done

# wait
# 固定 noise_embedding=0.1, 改变 ipc
for ipc in 1 10 20; do
    gpu=${gpus[$gpu_idx]}
    log_file="logs/AVE_ipc${ipc}_noise0.1.log"
    echo "Running on GPU $gpu with ipc=$ipc, noise_embedding=0.1 -> $log_file"
    CUDA_VISIBLE_DEVICES=$gpu python main_DM_AV_imagebind_cf_cos_random_noise.py \
        --dataset AVE \
        --Iteration 30 \
        --noise_embedding 0.1 \
        --num_eval 3 \
        --interval 1 \
        --ipc $ipc \
        > "$log_file" 2>&1 &
    gpu_idx=$(( (gpu_idx + 1) % ${#gpus[@]} ))
done

wait

# 固定 noise_embedding=0.1, 改变 ipc
for ipc in 1 10 20; do
    gpu=${gpus[$gpu_idx]}
    log_file="logs/VGG_ipc${ipc}_noise0.1.log"
    echo "Running on GPU $gpu with ipc=$ipc, noise_embedding=0.1 -> $log_file"
    CUDA_VISIBLE_DEVICES=$gpu python main_DM_AV_imagebind_cf_cos_random_noise.py \
        --dataset VGG_subset \
        --Iteration 30 \
        --noise_embedding 0.1 \
        --num_eval 3 \
        --interval 1 \
        --ipc $ipc \
        > "$log_file" 2>&1 &
    gpu_idx=$(( (gpu_idx + 1) % ${#gpus[@]} ))
done
wait


echo "✅ 所有实验已完成，日志保存在 logs/ 文件夹中。"
