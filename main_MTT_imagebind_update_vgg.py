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
import random
from reparam_module import ReparamModule
criterion = torch.nn.CrossEntropyLoss()
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)

def downscale(image_syn, scale_factor):
    image_syn = F.interpolate(image_syn, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    return image_syn

def resize_audio_for_imagebind(tensor):
    """Resize audio tensor to ImageBind standard dimensions (128, 204)"""
    if tensor is None:
        return None
    if len(tensor.shape) == 4:  # Already (B, C, H, W)
        return F.interpolate(tensor, size=(128, 204), mode='bilinear', align_corners=False)
    elif len(tensor.shape) == 3:  # (B, H, W)
        tensor = tensor.unsqueeze(1)  # (B, 1, H, W)
        return F.interpolate(tensor, size=(128, 204), mode='bilinear', align_corners=False)
    elif len(tensor.shape) == 2:  # (H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return F.interpolate(tensor, size=(128, 204), mode='bilinear', align_corners=False)
    else:
        raise ValueError(f"Unexpected audio tensor shape: {tensor.shape}")

def main(args):
    # 确保设备提前定义，避免后续使用问题
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {args.device}')
    
    print("--- WANDB DEBUG START ---")
    print(f"At start of main(), type of wandb is: {type(wandb)}")
    print(f"Attributes of wandb object: {dir(wandb)}")
    print(f"File path of wandb object: {getattr(wandb, '__file__', 'N/A or Built-in')}")
    print("--- WANDB DEBUG END ---")

    eval_it_pool = np.arange(0, args.Iteration+1, args.interval).tolist()
    channel, im_size, num_classes, mean, std, dst_train = get_train_dataset_origin(args.dataset, args)
    _, _, _, testloader = get_test_dataset_origin(args.dataset, args)
    args.cls_num = num_classes

    # Ensure correct dimensions for ImageBind (参考main_DM_AV_imagebind_lr_vgg1.py)
    im_size = list(im_size)
    im_size[1] = (224, 224)  # Image standard size for ImageBind
    im_size[0] = (128, 204)  # ImageBind audio standard size
    print('im_size[0] (audio):', im_size[0])
    print('im_size[1] (image):', im_size[1])


    accs_all_exps = []

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        log_dir = "wandb_logs"
        os.makedirs(log_dir, exist_ok=True)
        if args.wandb_disable:
            wandb.init(mode="disabled")
        else:
            wandb.init(sync_tensorboard=False,
                    project="AVDD",
                    config=args,
                    name = f'{args.id}_exp-{exp}') 
        base_seed = 178645
        seed = (base_seed + exp) % 100000
        set_seed(seed)

        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        labels_all = [dst_train[i]['label'] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        if args.init_herding:
            args.herd_path = get_herd_path(args.dataset)

        def get_aud_images_init(c, n): 
            if args.init_herding and args.dataset=='Music_21':        
                _, _, _, _, _, dst_train_center = get_train_dataset_origin('Music_21_center', args)
                print('Using herding indices....')
                with open(args.herd_path, 'rb') as f:
                    herd_idx_dict = pickle.load(f)
                idx_shuffle = herd_idx_dict[c]['av'][:n] 
                idx_aud = dst_train_center[idx_shuffle]['audio'].to(args.device)
                idx_aud = resize_audio_for_imagebind(idx_aud)
                idx_img = dst_train_center[idx_shuffle]['frame'].to(args.device)
                idx_img = torch.nn.functional.interpolate(idx_img, size=(224, 224), mode='bilinear', align_corners=False)
            else:
                idx_aud, idx_img = None, None
                if args.init_herding:
                    with open(args.herd_path, 'rb') as f:
                        herd_idx_dict = pickle.load(f)
                    idx_shuffle = herd_idx_dict[c]['av'][:n]
                elif len(indices_class[c]) < n:
                    idx_shuffle = np.random.permutation(indices_class[c])
                else:
                    idx_shuffle = np.random.permutation(indices_class[c])[:n]

                if args.input_modality == 'a' or args.input_modality == 'av':
                    idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)
                    idx_aud = resize_audio_for_imagebind(idx_aud)

                if args.input_modality == 'v' or args.input_modality == 'av':
                    idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
                    idx_img = torch.nn.functional.interpolate(idx_img, size=(224, 224), mode='bilinear', align_corners=False)

            return idx_aud, idx_img
        
        def get_aud_images(c, n): 
            idx_aud, idx_img = None, None
            idx_shuffle = np.random.permutation(indices_class[c])[:n].tolist()
            if args.input_modality == 'a' or args.input_modality == 'av':
                idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)
                idx_aud = resize_audio_for_imagebind(idx_aud)
            if args.input_modality == 'v' or args.input_modality == 'av':
                idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
                idx_img = torch.nn.functional.interpolate(idx_img, size=(224, 224), mode='bilinear', align_corners=False)
            return idx_aud, idx_img

        ''' initialize the synthetic data '''
        print("Loading expert trajectories...")
        expert_dir = os.path.join(args.buffer_path, args.dataset, args.model)

        if not os.path.exists(expert_dir):
            raise FileNotFoundError(f"Expert buffer path not found: {expert_dir}. Please run buffer.py first.")

        expert_files = [os.path.join(expert_dir, f) for f in os.listdir(expert_dir) if f.endswith(".pt")]
        if not expert_files:
            raise FileNotFoundError(f"No replay_buffer files found in {expert_dir}. Please run buffer.py first.")

        print(f"Found {len(expert_files)} expert files.")
        buffer = []
        for f in expert_files:
            print(f"Loading buffer file: {f}")
            buffer.extend(torch.load(f))
        print(f"Loaded a total of {len(buffer)} expert trajectories.")
        # +++ 新增代码结束 +++
        image_syn, aud_syn = None, None
        if args.input_modality == 'a' or args.input_modality == 'av':
            # 使用ImageBind标准音频尺寸128x204，注意保持4D格式
            aud_syn = torch.randn(size=(num_classes*args.ipc, 1, im_size[0][0], im_size[0][1]), dtype=torch.float, requires_grad=True, device=args.device)
        
        if args.input_modality == 'v' or args.input_modality == 'av':
            image_syn = torch.randn(size=(num_classes*args.ipc, channel[1], im_size[1][0], im_size[1][1]), dtype=torch.float, requires_grad=True, device=args.device)
        
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            if not args.idm_aug:
                aud_real_init, img_real_init = get_aud_images_init(c, args.ipc)
                
                if args.input_modality == 'a' or args.input_modality == 'av':
                    if aud_real_init is not None:
                        # 确保音频维度匹配ImageBind标准
                        aud_real_init = aud_real_init.detach().data
                        if len(aud_real_init.shape) == 3:  # (B, H, W)
                            aud_real_init = aud_real_init.unsqueeze(1)  # (B, 1, H, W)
                        # 调整尺寸到标准128x204
                        if aud_real_init.shape[-2:] != (im_size[0][0], im_size[0][1]):
                            aud_real_init = torch.nn.functional.interpolate(
                                aud_real_init, 
                                size=(im_size[0][0], im_size[0][1]), 
                                mode='bilinear', 
                                align_corners=False
                            )
                        aud_syn.data[c*args.ipc:(c+1)*args.ipc] = aud_real_init

                if args.input_modality == 'v' or args.input_modality == 'av':
                    if img_real_init is not None:
                        img_real_init = img_real_init.detach().data
                        # 调整图像尺寸到224x224
                        if img_real_init.shape[-2:] != (im_size[1][0], im_size[1][1]):
                            img_real_init = torch.nn.functional.interpolate(
                                img_real_init, 
                                size=(im_size[1][0], im_size[1][1]), 
                                mode='bilinear', 
                                align_corners=False
                            )
                        image_syn.data[c*args.ipc:(c+1)*args.ipc] = img_real_init
            else:
                for c in range(num_classes):
                    a_half_h, a_half_w = im_size[0][0]//2, im_size[0][1]//2
                    v_half_size = im_size[1][0]//2
                    auds_real, imgs_real = get_aud_images_init(c, args.ipc*args.idm_aug_count*args.idm_aug_count)
                    
                    # 处理音频数据 - 确保4D格式并调整尺寸
                    if auds_real is not None and (args.input_modality == 'a' or args.input_modality == 'av'):
                        auds_real = auds_real.detach().data
                        if len(auds_real.shape) == 3:  # (B, H, W)
                            auds_real = auds_real.unsqueeze(1)  # (B, 1, H, W)
                        # 确保尺寸匹配
                        if auds_real.shape[-2:] != (a_half_h*2, a_half_w*2):
                            auds_real = torch.nn.functional.interpolate(
                                auds_real, 
                                size=(a_half_h*2, a_half_w*2), 
                                mode='bilinear', 
                                align_corners=False
                            )
                        
                        start,end = 0, args.ipc
                        aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :a_half_h, :a_half_w] = downscale(auds_real[start:end], 0.5).data; start, end = end, end+args.ipc
                        aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, a_half_h:, :a_half_w] = downscale(auds_real[start:end], 0.5).data; start, end = end, end+args.ipc
                        aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :a_half_h, a_half_w:] = downscale(auds_real[start:end], 0.5).data; start, end = end, end+args.ipc
                        aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, a_half_h:, a_half_w:] = downscale(auds_real[start:end], 0.5).data; start, end = end, end+args.ipc

                    # 处理图像数据
                    if imgs_real is not None and (args.input_modality == 'v' or args.input_modality == 'av'):
                        imgs_real = imgs_real.detach().data
                        # 确保尺寸匹配
                        if imgs_real.shape[-2:] != (v_half_size*2, v_half_size*2):
                            imgs_real = torch.nn.functional.interpolate(
                                imgs_real, 
                                size=(v_half_size*2, v_half_size*2), 
                                mode='bilinear', 
                                align_corners=False
                            )
                        
                        v_half_size = im_size[1][0]//2; start,end = 0, args.ipc
                        image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :v_half_size, :v_half_size] = downscale(imgs_real[start:end], 0.5).data; start, end = end, end+args.ipc
                        image_syn.data[c*args.ipc:(c+1)*args.ipc, :, v_half_size:, :v_half_size] = downscale(imgs_real[start:end], 0.5).data; start, end = end, end+args.ipc
                        image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :v_half_size, v_half_size:] = downscale(imgs_real[start:end], 0.5).data; start, end = end, end+args.ipc
                        image_syn.data[c*args.ipc:(c+1)*args.ipc, :, v_half_size:, v_half_size:] = downscale(imgs_real[start:end], 0.5).data; start, end = end, end+args.ipc

        # ''' training '''
        params_to_optimize = [p for p in [aud_syn, image_syn] if p is not None]
        if not params_to_optimize:
            raise ValueError("没有可优化的参数 - 请检查input_modality设置")
        optimizer_img = torch.optim.SGD(params_to_optimize, lr=args.lr_img, momentum=0.5)

        # 初始化可学习的学习率，并为其创建优化器
        syn_lr = torch.tensor(args.lr_teacher).to(args.device).requires_grad_(True)
        optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5) 

        print('%s training begins'%get_time())
        for it in tqdm(range(args.Iteration+1)):

            if it in eval_it_pool:
                # if it == 0:
                #     continue
                ''' Evaluate synthetic data '''
                aud_syn_eval, image_syn_eval = None, None
                if args.input_modality == 'a' or args.input_modality == 'av':
                    aud_syn_eval = copy.deepcopy(aud_syn.detach())
                if args.input_modality == 'v' or args.input_modality == 'av':
                    image_syn_eval = copy.deepcopy(image_syn.detach())
                label_syn_eval = copy.deepcopy(label_syn.detach())
                
                aud_syn_eval1, image_syn_eval1 = aud_syn_eval, image_syn_eval
                if args.idm_aug:
                    if args.input_modality == 'a' or args.input_modality == 'av':
                        # 直接处理4D音频张量
                        aud_syn_eval1 = number_sign_augment(aud_syn_eval)
                    if args.input_modality == 'v' or args.input_modality == 'av':
                        image_syn_eval1 = number_sign_augment(image_syn_eval)
                    label_syn_eval = label_syn_eval.repeat(4)

                accs = []
                for it_eval in range(args.num_eval):
                    nets, net_eval = get_network_imagebind(args)
                    # 不要冻结任何参数，直接进行评估
                    acc = evaluate_synset_av(nets, net_eval, aud_syn_eval1, image_syn_eval1, label_syn_eval, testloader, args)
                    accs.append(acc)
                    print(f'it_eval: {it_eval} Val acc: {acc:.2f}%')

                wandb.log({'eval_acc': np.mean(accs)}, step=it)
                print(f'Mean eval at it: {it} Val acc: {np.mean(accs):.2f}%')

                if it == args.Iteration: # record the final results
                    accs_all_exps += accs
                
                ''' visualize and save '''
                if args.input_modality == 'a' or args.input_modality == 'av':
                    aud_syn_vis = copy.deepcopy(aud_syn_eval.detach().cpu())
                    torch.save(aud_syn_vis, args.syn_data_path + f'/exp_{exp}_audSyn_{it}.pt')
                    # grid = torchvision.utils.make_grid(aud_syn_vis, nrow=max(10, args.ipc), normalize=True, scale_each=True)
                    # wandb.log({"Synthetic_Audio": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                if args.input_modality == 'v' or args.input_modality == 'av':
                    image_syn_vis = copy.deepcopy(image_syn_eval.detach().cpu())
                    torch.save(image_syn_vis, args.syn_data_path + f'/exp_{exp}_imgSyn_{it}.pt')
                    for ch in range(channel[1]):
                        image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                    image_syn_vis[image_syn_vis<0] = 0.0
                    image_syn_vis[image_syn_vis>1] = 1.0
                    # grid = torchvision.utils.make_grid(image_syn_vis, nrow=max(10, args.ipc), normalize=True, scale_each=True)
                    # wandb.log({"Synthetic_Image": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

            # +++ 用以下代码块替换掉所有旧的训练逻辑 +++

            # --- MTT 核心训练模块 ---
            wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

            # 1. 创建学生网络，并用ReparamModule包裹
            _, student_net = get_network_imagebind(args)
            # 假设 get_network_imagebind 返回的 net_eval 是一个完整的 nn.Module
            student_net = ReparamModule(student_net).to(args.device)
            student_net.train()
            num_params = sum([np.prod(p.size()) for p in student_net.parameters()])

            # 2. 从buffer中随机采样一条专家轨迹片段
            expert_trajectory = random.choice(buffer)
            start_epoch = random.randint(0, args.max_start_epoch)
            starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in expert_trajectory[start_epoch]], 0)
            target_params   = torch.cat([p.data.to(args.device).reshape(-1) for p in expert_trajectory[start_epoch + args.expert_epochs]], 0)
                        # 3. 初始化学生网络参数
            student_params = [starting_params.requires_grad_(True)]

            # 4. 在合成数据上模拟训练过程
            for step in range(args.syn_steps):
                indices = torch.randperm(len(label_syn))
                idx_syn = indices[:args.batch_syn]

                y_hat = label_syn[idx_syn].to(args.device)
                x_aud_syn, x_img_syn = None, None
                if 'a' in args.input_modality:
                    x_aud_syn = aud_syn[idx_syn]
                if 'v' in args.input_modality:
                    x_img_syn = image_syn[idx_syn]

                output = student_net(x_aud_syn, x_img_syn, flat_param=student_params[-1])

                ce_loss = criterion(output, y_hat)
                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
                student_params.append(student_params[-1] - syn_lr * grad)

            # 5. 计算轨迹损失
            param_loss = torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
            grand_loss = param_loss / param_dist if param_dist > 0 else param_loss

            # 6. 优化合成数据和合成学习率
            optimizer_img.zero_grad()
            optimizer_lr.zero_grad()
            grand_loss.backward()
            optimizer_img.step()
            optimizer_lr.step()

            wandb.log({"Grand_Loss": grand_loss.detach().cpu(), "Start_Epoch": start_epoch}, step=it)

            del student_params # 清理内存

            if it % 10 == 0:
                print(f'{get_time()} iter = {it:04d}, loss = {grand_loss.item():.4f}')

    print('\n==================== Final Results ====================\n')
    accs = accs_all_exps
    print('Run %d experiments, random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, len(accs), np.mean(accs), np.std(accs)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # --- 核心训练参数 ---
    parser.add_argument('--dataset', type=str, default='VGG_subset', help='dataset')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--Iteration', type=int, default=5000, help='training iterations')
    parser.add_argument('--num_exp', type=int, default=3, help='the number of experiments')    

    # --- 评估参数 ---
    parser.add_argument('--epoch_eval_train', type=int, default=30, help='epochs to train a model with synthetic data')
    parser.add_argument('--interval', type=int, default=1000, help='interval to evaluate the synthetic data')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for adam optimizer')

    # --- 模型架构参数 ---
    parser.add_argument('--arch_frame', type=str, default='imagebind_huge', help='ImageBind model architecture')
    parser.add_argument('--arch_classifier', type=str, default='imagebind_head', help='classifier architecture')
    parser.add_argument('--weights_sound', type=str, default='', help='weights for sound network')
    parser.add_argument('--weights_frame', type=str, default='', help='weights for frame network')
    parser.add_argument('--weights_classifier', type=str, default='', help='weights for classifier')

    # --- 数据与批次参数 ---
    parser.add_argument('--data_path', type=str, default='data', help='path to dataset')
    parser.add_argument('--batch_syn', type=int, default=32, help='batch size for syn data')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--input_modality', type=str, default='av', help='a/v/av')

    # --- 数据初始化与增强 ---
    parser.add_argument('--init_herding', action='store_true', help='init using herding or not')
    parser.add_argument('--idm_aug', action='store_true', help='use image-divisible-mixing augmentation')
    parser.add_argument('--idm_aug_count', type=int, default=2, help='number of images per image during IDM')

    # --- MTT 核心参数 ---
    parser.add_argument('--model', type=str, default='AV-ConvNet', help='model architecture name for buffer path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='Path to load expert trajectory buffers')
    parser.add_argument('--expert_epochs', type=int, default=3, help='How many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='How many steps to take on synthetic data in inner loop')
    parser.add_argument('--max_start_epoch', type=int, default=5, help='Max epoch we can start at')

    # --- MTT 学习率参数 ---
    parser.add_argument('--lr_img', type=float, default=1.0, help='Learning rate for updating synthetic data')
    parser.add_argument('--lr_lr', type=float, default=1e-5, help='Learning rate for updating synthetic learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='Initialization for synthetic learning rate')
    parser.add_argument('--dsa_strategy', type=str, default='none', help='differentiable Siamese augmentation strategy')

    # --- MTT 核心参数 ---
    # --- 其他 ---
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--wandb_disable', action='store_true', help='use to disable wandb logging')
    parser.add_argument('--base_syn_data_dir', type=str, default='data/syn_data_train', help='base directory to save synthetic data')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser.add_argument("--deepspeed", action="store_true",
                        help="Enable deepspeed")
    args = parser.parse_args()
    
    # --- 在代码中自动设置的参数 ---
    args.dsa_param = ParamDiffAug() # 尽管dsa_strategy已删除,为防其他utils函数可能用到,可保留此行
    args.dsa = False # MTT不使用DSA,直接设为False

    args.lr_steps_interval = 10
    args.lr_steps = np.arange(args.lr_steps_interval, args.epoch_eval_train, args.lr_steps_interval).tolist()

    args.id = f'{args.dataset}_ipc-{args.ipc}'
    args.syn_data_path = os.path.join(args.base_syn_data_dir, args.id)
    os.makedirs(args.syn_data_path, exist_ok=True)
    
    main(args)


