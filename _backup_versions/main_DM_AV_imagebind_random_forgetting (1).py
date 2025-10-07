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
import time
from nets.imagebind.models.imagebind_model import ModalityType
from utils.data_utils import get_test_dataset_origin,  get_train_dataset_origin, get_herd_path, \
    ParamDiffAug, number_sign_augment, DiffAugment, get_time
from utils.train_utils_DM import evaluate_synset_av, get_network_imagebind
from tqdm import tqdm
warnings.filterwarnings("ignore")
torch.set_num_threads(8)
import time

# ==================== 在此处粘贴以下全新代码 ====================
def run_forgetting_analysis(model, full_dataset, device, analysis_epochs=10, lr=0.01):
    """
    执行Forgetting分析，返回每个样本的“忘记次数”。
    忘记次数越少，代表样本越“核心”。
    """
    print("🚀 Starting Forgetting analysis...")
    
    n_samples = len(full_dataset)
    # 初始化每个样本的忘记次数为0
    forgetting_counts = np.zeros(n_samples, dtype=np.uint32)
    # 记录上一个epoch的预测正确性 (-1: 未知, 0: 错误, 1: 正确)
    last_epoch_correctness = np.full(n_samples, -1, dtype=np.int8)

    # 创建一个临时的优化器和数据加载器用于分析
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # 注意：分析时 shuffle 必须为 False，以保证样本顺序固定
    loader = torch.utils.data.DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=4)

    for epoch in range(analysis_epochs):
        model.train()
        current_epoch_correctness = np.full(n_samples, -1, dtype=np.int8)
        
        pbar = tqdm(loader, desc=f"Forgetting Analysis Epoch {epoch+1}/{analysis_epochs}")
        
        # 记录当前样本在loader中的偏移量
        offset = 0
        for inputs, targets in pbar:
            # 将数据移动到设备
            targets_gpu = targets.to(device)
            if isinstance(inputs, dict):
                for key, value in inputs.items():
                    inputs[key] = value.to(device)
            else:
                inputs = inputs.to(device)

            # 训练模型
            optimizer.zero_grad()
            outputs = model(inputs)
            # 对于ImageBind的字典输出，我们取最终的logits
            if isinstance(outputs, dict):
                outputs = outputs[ModalityType.VISION] # 或者一个融合后的logits，取决于模型实现
            loss = F.cross_entropy(outputs, targets_gpu)
            loss.backward()
            optimizer.step()

            # 记录当前批次的预测正确性
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                correct_flags = (preds == targets_gpu).cpu().numpy()
                batch_size = targets.size(0)
                current_epoch_correctness[offset : offset + batch_size] = correct_flags
                offset += batch_size

        # 比较上一个和当前epoch，统计忘记事件
        if last_epoch_correctness[0] != -1: # 从第二个epoch开始统计
            # 忘记事件：从正确(1)变为错误(0)
            forgotten_mask = (last_epoch_correctness == 1) & (current_epoch_correctness == 0)
            forgetting_counts[forgotten_mask] += 1
            
        # 更新状态以备下一个epoch使用
        last_epoch_correctness = current_epoch_correctness
        
        pbar.set_postfix({"Total Forgotten Events": np.sum(forgetting_counts)})

    print(f"✅ Forgetting analysis finished. Total forgotten events: {np.sum(forgetting_counts)}")
    return forgetting_counts
# =================================================================

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)

def downscale(image_syn, scale_factor):
    image_syn = F.upsample(image_syn, scale_factor=scale_factor, mode='bilinear')
    return image_syn

def resize_to_square(tensor, target_size=224, mode='bilinear'):
    """Resize tensor to square dimensions for ImageBind compatibility"""
    if tensor is None:
        return None
    # Handle different tensor formats
    if len(tensor.shape) == 4:  # (B, C, H, W)
        return F.interpolate(tensor, size=(target_size, target_size), mode=mode, align_corners=False)
    elif len(tensor.shape) == 5:  # (B, C, D, H, W) - for audio
        # Resize spatial dimensions, keep depth
        return F.interpolate(tensor.squeeze(2), size=(target_size, target_size), mode=mode, align_corners=False).unsqueeze(2)
    return tensor

def resize_audio_for_imagebind(tensor):
    """Resize audio tensor to ImageBind standard dimensions (128, 204)"""
    if tensor is None:
        return None
    return F.interpolate(tensor, size=(128, 204), mode='bilinear', align_corners=False)

def main(args):
    eval_it_pool = np.arange(0, args.Iteration+1, args.interval).tolist()
    channel, im_size, num_classes, mean, std, dst_train = get_train_dataset_origin(args.dataset, args)
    _, _, _, testloader = get_test_dataset_origin(args.dataset, args)
    args.cls_num = num_classes

    # Ensure correct dimensions for ImageBind
    im_size = list(im_size)
    im_size[1] = (224, 224)  # Image standard size
    im_size[0] = (128, 204)  # ImageBind audio standard size
    print('im_size[0] (audio):', im_size[0])
    print('im_size[1] (image):', im_size[1])


    accs_all_exps = []

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        
        if args.wandb_disable:
            wandb.init(mode="disabled")
        else:
            wandb.init(sync_tensorboard=False,
                    project="AVDD_Coreset", # (可选) 改个新项目名
                    config=args,
                    name = f'{args.id}_method-{args.selection_method}_exp-{exp}')

        base_seed = 178645
        seed = (base_seed + exp) % 100000
        set_seed(seed)
        
        forgetting_scores_sorted_indices = None
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        labels_all = [dst_train[i]['label'] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        # ==================== START OF MODIFICATION 2 ====================
        # 根据新的参数设置 init_herding
        if args.selection_method == 'herding':
            args.init_herding = True
            args.herd_path = get_herd_path(args.dataset)
            print("Selection method: Herding")
        else:
            args.init_herding = False
            print("Selection method: Random")
            
    def get_aud_images_init(c, n):
        # 初始化索引列表，稍后根据不同方法填充
        idx_shuffle = []
        
        # 1. Music_21数据集的Herding是一个特殊情况，优先处理
        if args.init_herding and args.dataset == 'Music_21':
            print('Using herding indices for Music_21...')
            # (这部分特殊逻辑保持不变)
            _, _, _, _, _, dst_train_center = get_train_dataset_origin('Music_21_center', args)
            try:
                with open(args.herd_path, 'rb') as f:
                    herd_idx_dict = pickle.load(f)
                idx_shuffle = herd_idx_dict[c]['av'][:min(n, len(herd_idx_dict[c]['av']))]
                idx_aud = dst_train_center[idx_shuffle]['audio'].to(args.device)
                idx_img = dst_train_center[idx_shuffle]['frame'].to(args.device)
                return idx_aud, idx_img # 特殊情况直接返回
            except (FileNotFoundError, KeyError, IndexError) as e:
                print(f"Warning: Herding for Music_21 failed, falling back to random. Reason: {e}")
                # 如果失败，则清空idx_shuffle，交由后续的随机逻辑处理
                idx_shuffle = []
    
        # 2. 对于所有其他情况，先根据方法确定索引 (idx_shuffle)
        # --------------------------------------------------------------------
        # A. Herding 方法 (非 Music_21 数据集)
        if args.selection_method == 'herding' and not idx_shuffle:
            print(f'Using Herding indices for class {c}...')
            try:
                with open(args.herd_path, 'rb') as f:
                    herd_idx_dict = pickle.load(f)
                idx_shuffle = herd_idx_dict[c]['av'][:min(n, len(herd_idx_dict[c]['av']))]
            except Exception as e:
                print(f"Warning: Herding failed, falling back to random. Reason: {e}")
                idx_shuffle = [] # 失败则清空，后续会走随机逻辑
    
        # B. Forgetting 方法
        elif args.selection_method == 'forgetting':
            print(f'Using Forgetting indices for class {c}...')
            # 从预先计算好的、已排序的索引中直接取出“忘记次数最少”的前n个
            if forgetting_scores_sorted_indices:
                 idx_shuffle = forgetting_scores_sorted_indices[c][:min(n, len(forgetting_scores_sorted_indices[c]))]
            else:
                 print("Warning: Forgetting scores not found, falling back to random.")
                 idx_shuffle = [] # 失败则清空
    
        # C. Random 方法 (作为默认或备用选项)
        if not idx_shuffle: # 如果经过以上方法后idx_shuffle仍为空，则使用随机
            print(f'Using Random indices for class {c}...')
            if len(indices_class[c]) < n:
                idx_shuffle = np.random.permutation(indices_class[c])
            else:
                idx_shuffle = np.random.permutation(indices_class[c])[:min(n, len(indices_class[c]))]
    
        # 3. 使用上面确定的索引 (idx_shuffle) 来统一提取数据
        # --------------------------------------------------------------------
        idx_aud, idx_img = None, None
        try:
            if args.input_modality == 'a' or args.input_modality == 'av':
                idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)
            if args.input_modality == 'v' or args.input_modality == 'av':
                idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
        except (IndexError, KeyError) as e:
            print(f"Warning: Data fetching failed for class {c} with selected indices. Error: {e}")
            # 如果出错，确保返回None
            idx_aud, idx_img = None, None
    
        return idx_aud, idx_img
        
        def get_aud_images(c, n): 
            idx_aud, idx_img = None, None
            idx_shuffle = np.random.permutation(indices_class[c])[:n].tolist()
            if args.input_modality == 'a' or args.input_modality == 'av':
                idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)
            if args.input_modality == 'v' or args.input_modality == 'av':
                idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
            return idx_aud, idx_img

        ''' initialize the synthetic data '''
        image_syn, aud_syn = None, None
        if args.input_modality == 'a' or args.input_modality == 'av':
           # [修正] 使用ImageBind标准的音频尺寸 128x204
           aud_syn = torch.randn(size=(num_classes*args.ipc, 1, 128, 204), dtype=torch.float, requires_grad=True, device=args.device)
        
        if args.input_modality == 'v' or args.input_modality == 'av':
            # 视频尺寸保持 224x224，正确
            image_syn = torch.randn(size=(num_classes*args.ipc, channel[1], 224, 224), dtype=torch.float, requires_grad=True, device=args.device)
        
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            if not args.idm_aug:
                aud_real_init, img_real_init = get_aud_images_init(c, args.ipc)
                
                # --- 音频初始化修正 ---
                if aud_real_init is not None and (args.input_modality == 'a' or args.input_modality == 'av'):
                    # 确保原始音频数据有通道维度
                    if aud_real_init.dim() == 3: # Shape: (ipc, H, W)
                        aud_real_init = aud_real_init.unsqueeze(1) # Becomes (ipc, 1, H, W)

                    # [关键] 将加载的真实音频缩放到ImageBind标准的128x204尺寸
                    aud_real_init_resized = F.interpolate(aud_real_init, size=(128, 204), mode='bilinear', align_corners=False)

                    # 安全地赋值
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc] = aud_real_init_resized.detach().data

                # --- 视频初始化 ---
                if img_real_init is not None and (args.input_modality == 'v' or args.input_modality == 'av'):
                    # (可选) 如果视频尺寸不匹配，也进行缩放
                    img_real_init_resized = F.interpolate(img_real_init, size=(224, 224), mode='bilinear', align_corners=False)
                    image_syn.data[c*args.ipc:(c+1)*args.ipc] = img_real_init_resized.detach().data
            else:
                for c in range(num_classes):
                    a_half_h, a_half_w = im_size[0][0]//2, im_size[0][1]//2
                    v_half_size = im_size[1][0]//2
                    auds_real, imgs_real = get_aud_images_init(c, args.ipc*args.idm_aug_count*args.idm_aug_count)
                    
                    start,end = 0, args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :a_half_h, :a_half_w] = downscale(auds_real[start:end], 0.5).data; start, end = end, end+args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, a_half_h:, :a_half_w] = downscale(auds_real[start:end], 0.5).data; start, end = end, end+args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :a_half_h, a_half_w:] = downscale(auds_real[start:end], 0.5).data; start, end = end, end+args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, a_half_h:, a_half_w:] = downscale(auds_real[start:end], 0.5).data; start, end = end, end+args.ipc

                    v_half_size = im_size[1][0]//2; start,end = 0, args.ipc
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :v_half_size, :v_half_size] = downscale(imgs_real[start:end], 0.5).data; start, end = end, end+args.ipc
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, v_half_size:, :v_half_size] = downscale(imgs_real[start:end], 0.5).data; start, end = end, end+args.ipc
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :v_half_size, v_half_size:] = downscale(imgs_real[start:end], 0.5).data; start, end = end, end+args.ipc
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, v_half_size:, v_half_size:] = downscale(imgs_real[start:end], 0.5).data; start, end = end, end+args.ipc


        # ''' training '''
# ==================== 在此处粘贴以下全新代码 ====================
        # 根据新的参数设置 init_herding
        if args.selection_method == 'herding':
            args.init_herding = True
            args.herd_path = get_herd_path(args.dataset)
            print("Selection method: Herding")
        else:
            args.init_herding = False
            print("Selection method: Random")

        ''' 步骤 1: 使用指定方法选择并准备数据子集 (Coreset) '''
        # 这部分逻辑借用了原始代码的初始化部分
        print(f'Starting coreset selection with method: {args.selection_method}')
        
        # 将 requires_grad 设置为 False，因为我们不优化这些数据
        if image_syn is not None: image_syn.requires_grad_(False)
        if aud_syn is not None: aud_syn.requires_grad_(False)

        # 这里的填充逻辑与原始代码相同，get_aud_images_init 会根据 args.init_herding 的值自动选择方法
        for c in range(num_classes):
            aud_real_init, img_real_init = get_aud_images_init(c, args.ipc)
            
            if aud_real_init is not None and (args.input_modality == 'a' or args.input_modality == 'av'):
                if aud_real_init.dim() == 3:
                    aud_real_init = aud_real_init.unsqueeze(1)
                aud_real_init_resized = F.interpolate(aud_real_init, size=(128, 204), mode='bilinear', align_corners=False)
                aud_syn.data[c*args.ipc:(c+1)*args.ipc] = aud_real_init_resized.detach().data

            if img_real_init is not None and (args.input_modality == 'v' or args.input_modality == 'av'):
                img_real_init_resized = F.interpolate(img_real_init, size=(224, 224), mode='bilinear', align_corners=False)
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = img_real_init_resized.detach().data
        
        print("Coreset selection finished.")


        ''' 步骤 2: 直接评估选出的 Coreset '''
        # 这部分代码是从原始的 if it in eval_it_pool: 块中提取并修改的
        print("Starting evaluation of the selected coreset...")
        
        aud_syn_eval, image_syn_eval = aud_syn, image_syn
        label_syn_eval = label_syn
        
        accs = []
        for it_eval in range(args.num_eval):
            nets, net_eval = get_network_imagebind(args)
            # 调用评估函数
            acc = evaluate_synset_av(nets, net_eval, aud_syn_eval, image_syn_eval, label_syn_eval, testloader, args)
            accs.append(acc)
            print(f'Eval run {it_eval+1}/{args.num_eval}: Val acc: {acc:.2f}%')

        # 记录本次实验 (exp) 的平均结果
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        wandb.log({'eval_acc_mean': mean_acc, 'eval_acc_std': std_acc})
        print(f'Mean validation accuracy for Exp {exp}: {mean_acc:.2f}% ± {std_acc:.2f}%')

        # 将本次实验的所有评估结果存入总列表
        accs_all_exps.extend(accs)
        
        # 保存选出的 coreset，方便复现
        coreset_save_dir = os.path.join(args.syn_data_path, 'coresets')
        os.makedirs(coreset_save_dir, exist_ok=True)
        if args.input_modality == 'a' or args.input_modality == 'av':
            torch.save(aud_syn.cpu(), os.path.join(coreset_save_dir, f'exp{exp}_{args.selection_method}_ipc{args.ipc}_aud.pt'))
        if args.input_modality == 'v' or args.input_modality == 'av':
            torch.save(image_syn.cpu(), os.path.join(coreset_save_dir, f'exp{exp}_{args.selection_method}_ipc{args.ipc}_img.pt'))

        print('Finished experiment run %d.' % exp)
# =================================================================
        wandb.finish()

    print('\n==================== Final Results ====================\n')
    accs = accs_all_exps
    print('Run %d experiments, random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, len(accs), np.mean(accs), np.std(accs)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    #training parameters
    parser.add_argument('--dataset', type=str, default='VGG_subset', help='dataset')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--Iteration', type=int, default=5000, help='training iterations')
    parser.add_argument('--num_exp', type=int, default=3, help='the number of experiments')    

    parser.add_argument('--lam_cgm', type=float, default=10.0, help='weight for cross-modal gap matching loss')
    parser.add_argument('--lam_icm', type=float, default=10.0, help='weight for implicit cross matching loss')
    parser.add_argument('--lr_syn_aud', type=float, default=0.05, help='learning rate for updating synthetic audio specs')
    parser.add_argument('--lr_syn_img', type=float, default=0.05, help='learning rate for updating synthetic image')
    
    #evaluation parameters
    parser.add_argument('--epoch_eval_train', type=int, default=30, help='epochs to train a model with synthetic data')
    parser.add_argument('--interval', type=int, default=1000, help='interval to evaluate the synthetic data')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')

    parser.add_argument('--arch_sound', type=str, default='convNet', help='convNet')
    parser.add_argument('--weights_sound', type=str, default='', help='weights for sound')   
    parser.add_argument('--arch_frame', type=str, default='convNet', help='convNet')
    parser.add_argument('--weights_frame', type=str, default='', help='weights for frame')
    parser.add_argument('--arch_classifier', type=str, default='ensemble', help='ensemble')
    parser.add_argument('--weights_classifier', type=str, default='', help='weights for classifier')

    parser.add_argument('--lr_frame', type=float, default=1e-4, help='learning rate for updating synthetic audio specs')
    parser.add_argument('--lr_sound', type=float, default=1e-3, help='sound learning rate')    
    parser.add_argument('--lr_classifier', type=float, default=1e-3, help='classifier learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='classifier learning rate')
    parser.add_argument('--batch_syn', type=int, default=32, help='batch size for syn data')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')

    #data parameters
    parser.add_argument('--init_herding', action='store_true', help='init using herding or not')
    parser.add_argument('--input_modality', type=str, default='av', help='a/v/av')
    parser.add_argument('--idm_aug_count', type=int, default=2, help='number of images per image during IDM')
    parser.add_argument('--idm_aug', action='store_true', help='use Augmentation or not')
    parser.add_argument('--wandb_disable', action='store_false', help='wandb disable')
    parser.add_argument('--batch_real', type=int, default=128, help='batch size for real data')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--selection_method', type=str, default='random', choices=['random', 'herding', 'forgetting'], help='Coreset selection method')
    parser.add_argument('--base_syn_data_dir', type=str, default='data/syn_data_train', help='a/v/av')
    
    ### CF arguments
    

    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.lr_steps_interval = 10
    args.lr_steps = np.arange(args.lr_steps_interval, args.epoch_eval_train, args.lr_steps_interval).tolist()

    args.id = f'{args.dataset}_ipc-{args.ipc}'
    args.syn_data_path = os.path.join(args.base_syn_data_dir, args.id)
    os.makedirs(args.syn_data_path, exist_ok=True)
    
    main(args)


