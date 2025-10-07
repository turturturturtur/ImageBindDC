import os
import copy
import argparse
import torchvision
import wandb
import torch
import numpy as np
import pickle
import warnings
import torch.nn.functional as F
import logging
from utils.data_utils import ParamDiffAug, get_test_dataset_origin, get_train_dataset_origin, get_herd_path
from utils.train_utils_DM import evaluate_synset_av, get_network_imagebind
from tqdm import tqdm
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
torch.set_num_threads(8)

def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)

def main(args):
    # --- 1. 初始化和数据加载 (与原版类似) ---
    channel, _, num_classes, _, _, dst_train = get_train_dataset_origin(args.dataset, args)
    _, _, _, testloader = get_test_dataset_origin(args.dataset, args)
    args.cls_num = num_classes

    # [关键修正] 为适配ImageBind和避免尺寸错误，硬编码目标尺寸
    image_target_size = (224, 224)
    audio_target_size = (128, 204)
    print(f'Target image size: {image_target_size}')
    print(f'Target audio size: {audio_target_size}')

    accs_all_exps = []
    for exp in range(args.num_exp):
        print(f'\n================== Exp {exp} ==================\n')
        
        if args.wandb_disable:
            wandb.init(mode="disabled")
        else:
            wandb.init(project="Coreset_Evaluation",
                       config=args,
                       name=f'{args.dataset}_ipc{args.ipc}_method-{args.selection_method}_exp-{exp}')

        set_seed(exp) # 使用实验编号作为种子

        labels_all = [dst_train[i]['label'] for i in range(len(dst_train))]
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        for c in range(num_classes):
            print(f'Class c = {c}: {len(indices_class[c])} real images')

        # --- 2. 根据命令行参数选择数据选择方法 (Herding 或 Random) ---
        if args.selection_method == 'herding':
            args.init_herding = True
            try:
                args.herd_path = get_herd_path(args.dataset)
                # 确认herding文件存在
                if not os.path.exists(args.herd_path):
                    raise FileNotFoundError
                print(f"Selection method: Herding (using path: {args.herd_path})")
            except (FileNotFoundError, KeyError):
                print(f"Warning: Herding file not found for dataset {args.dataset}. Falling back to Random selection.")
                args.init_herding = False
                args.selection_method = 'random'
        else:
            args.init_herding = False
            print("Selection method: Random")
            
        def get_aud_images_init(c, n):
            """根据init_herding标志选择herding或random索引来获取数据"""
            idx_aud, idx_img = None, None
            
            if args.init_herding:
                with open(args.herd_path, 'rb') as f:
                    herd_idx_dict = pickle.load(f)
                # 安全地获取索引，防止数量不足
                num_available = len(herd_idx_dict[c]['av'])
                idx_shuffle = herd_idx_dict[c]['av'][:min(n, num_available)]
            else: # Random
                num_available = len(indices_class[c])
                idx_shuffle = np.random.permutation(indices_class[c])[:min(n, num_available)]

            if not isinstance(idx_shuffle, list):
                idx_shuffle = idx_shuffle.tolist()

            # 从数据集中提取数据
            if 'a' in args.input_modality:
                idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)
            if 'v' in args.input_modality:
                idx_img = dst_train[idx_shuffle]['frame'].to(args.device)

            return idx_aud, idx_img
        
        # --- 3. 创建Coreset: 初始化空的合成数据张量，并用选择的真实数据填充 ---
        print(f'Starting coreset selection with method: {args.selection_method}')
        
        image_coreset, audio_coreset = None, None
        if 'a' in args.input_modality:
           audio_coreset = torch.randn(size=(num_classes*args.ipc, 1, *audio_target_size), dtype=torch.float, device=args.device)
        if 'v' in args.input_modality:
            image_coreset = torch.randn(size=(num_classes*args.ipc, channel[1], *image_target_size), dtype=torch.float, device=args.device)
        
        label_coreset = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, device=args.device).view(-1)

        for c in range(num_classes):
            aud_real_init, img_real_init = get_aud_images_init(c, args.ipc)
            
            start_idx, end_idx = c * args.ipc, (c + 1) * args.ipc

            # [关键修正] 填充时强制将真实数据缩放到目标尺寸
            if aud_real_init is not None and 'a' in args.input_modality:
                if aud_real_init.dim() == 5: aud_real_init = aud_real_init.squeeze(2)
                if aud_real_init.dim() == 3: aud_real_init = aud_real_init.unsqueeze(1)
                aud_real_resized = F.interpolate(aud_real_init, size=audio_target_size, mode='bilinear', align_corners=False)
                audio_coreset.data[start_idx:end_idx] = aud_real_resized.detach().data

            if img_real_init is not None and 'v' in args.input_modality:
                img_real_resized = F.interpolate(img_real_init, size=image_target_size, mode='bilinear', align_corners=False)
                image_coreset.data[start_idx:end_idx] = img_real_resized.detach().data

        print("Coreset selection finished.")

        # --- 4. 直接评估选出的 Coreset (不再有训练循环) ---
        print("Starting evaluation of the selected coreset...")
        
        # Coreset就是我们的评估对象，不需要拷贝
        aud_eval, img_eval, lab_eval = audio_coreset, image_coreset, label_coreset
        
        accs = []
        for it_eval in range(args.num_eval):
            # 评估函数会创建全新的、可训练的网络
            nets, net_eval = get_network_imagebind(args)
            acc = evaluate_synset_av(nets, net_eval, aud_eval, img_eval, lab_eval, testloader, args)
            accs.append(acc)
            print(f'Eval run {it_eval+1}/{args.num_eval}: Val acc: {acc:.2f}%')

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f'Mean validation accuracy for Exp {exp}: {mean_acc:.2f}% ± {std_acc:.2f}%')
        wandb.log({'eval_acc_mean': mean_acc, 'eval_acc_std': std_acc})

        accs_all_exps.extend(accs)
        
        # --- 5. 保存选出的coreset ---
        coreset_save_dir = os.path.join(args.base_syn_data_dir, args.id, 'coresets')
        os.makedirs(coreset_save_dir, exist_ok=True)
        if audio_coreset is not None:
            torch.save(audio_coreset.cpu(), os.path.join(coreset_save_dir, f'exp{exp}_{args.selection_method}_ipc{args.ipc}_aud.pt'))
        if image_coreset is not None:
            torch.save(image_coreset.cpu(), os.path.join(coreset_save_dir, f'exp{exp}_{args.selection_method}_ipc{args.ipc}_img.pt'))

        print(f'Finished experiment run {exp}.')
        wandb.finish()

    print('\n==================== Final Results ====================\n')
    final_mean = np.mean(accs_all_exps)
    final_std = np.std(accs_all_exps)
    print(f'Final results over {args.num_exp} experiments ({len(accs_all_exps)} total runs):')
    print(f'Mean Accuracy = {final_mean:.2f}%')
    print(f'Std Deviation = {final_std:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Parameter Set')
    
    # ===== 数据集与模态 =====
    parser.add_argument('--dataset',         type=str, default='AVE', help='AVE / VGG_subset / etc.')
    parser.add_argument('--ipc',             type=int, default=1,     help='items/images per class')
    parser.add_argument('--input_modality',  type=str, default='av',  choices=['a', 'v', 'av'], help='a/v/av')
    
    # ===== 蒸馏 / 合成阶段 =====
    parser.add_argument('--selection_method', type=str, default='random', choices=['random', 'herding'])
    parser.add_argument('--init_herding',     action='store_true', help='init coreset with herding')
    parser.add_argument('--Iteration',        type=int, default=5000, help='distillation iterations')
    parser.add_argument('--interval',         type=int, default=1000, help='eval interval during distillation')
    parser.add_argument('--lr_syn_aud',       type=float, default=0.05, help='lr for synthetic audio')
    parser.add_argument('--lr_syn_img',       type=float, default=0.05, help='lr for synthetic image')
    parser.add_argument('--lam_icm',          type=float, default=10.0, help='AVDD ICM weight')
    parser.add_argument('--lam_cgm',          type=float, default=10.0, help='AVDD CGM weight')
    
    # ===== 评估阶段 =====
    parser.add_argument('--num_exp',          type=int, default=3,   help='number of experiments')
    parser.add_argument('--num_eval',         type=int, default=5,   help='models per experiment')
    parser.add_argument('--epoch_eval_train', type=int, default=30,  help='epochs to train model with coreset/syn data')
    
    # ===== 网络结构 =====
    parser.add_argument('--arch_frame',       type=str, default='imagebind_huge')
    parser.add_argument('--arch_sound',       type=str, default='imagebind_huge')
    parser.add_argument('--arch_classifier',  type=str, default='ensemble')
    
    # ===== 权重文件 =====
    parser.add_argument('--weights_frame',     type=str, default='')
    parser.add_argument('--weights_sound',     type=str, default='')
    parser.add_argument('--weights_classifier',type=str, default='')
    
    # ===== 训练超参 =====
    parser.add_argument('--lr_frame',        type=float, default=1e-4)
    parser.add_argument('--lr_sound',        type=float, default=1e-3)
    parser.add_argument('--lr_classifier',   type=float, default=1e-3)
    parser.add_argument('--weight_decay',    type=float, default=1e-4)
    parser.add_argument('--beta1',           type=float, default=0.9)
    
    # ===== 批次大小 =====
    parser.add_argument('--batch_syn',       type=int, default=32, help='batch size for syn/coreset')
    parser.add_argument('--batch_real',      type=int, default=128, help='batch size for real data')
    
    # ===== 数据增强 =====
    parser.add_argument('--idm_aug',         action='store_true')
    parser.add_argument('--idm_aug_count',   type=int, default=2)
    parser.add_argument('--dsa_strategy',    type=str, default='none')
    
    # ===== 存储与日志 =====
    parser.add_argument('--base_syn_data_dir', type=str, default='data/coreset_data')
    parser.add_argument('--num_workers',       type=int, default=4)
    parser.add_argument('--wandb_disable',     action='store_true')
    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.id = f'{args.dataset}_ipc-{args.ipc}'
    args.lr_steps_interval = 10
    args.lr_steps = np.arange(args.lr_steps_interval, args.epoch_eval_train, args.lr_steps_interval).tolist()

    args.id = f'{args.dataset}_ipc-{args.ipc}'
    args.syn_data_path = os.path.join(args.base_syn_data_dir, args.id)
    os.makedirs(args.syn_data_path, exist_ok=True)

    main(args)