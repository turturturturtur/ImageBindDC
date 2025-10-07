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


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)

def downscale(image_syn, scale_factor):
    image_syn = F.upsample(image_syn, scale_factor=scale_factor, mode='bilinear')
    return image_syn

def main(args):
    eval_it_pool = np.arange(0, args.Iteration+1, args.interval).tolist()
    channel, im_size, num_classes, mean, std, dst_train = get_train_dataset_origin(args.dataset, args)
    _, _, _, testloader = get_test_dataset_origin(args.dataset, args)
    args.cls_num = num_classes

    square_size = min(im_size[1][0], im_size[1][1])
    im_size = list(im_size)  # 先把im_size整体转成list以便赋值
    im_size[1] = (square_size, square_size)
    print('im_size[1] after enforcing square:', im_size[1])


    accs_all_exps = []

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        
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
                idx_img = dst_train_center[idx_shuffle]['frame'].to(args.device)
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

                if args.input_modality == 'v' or args.input_modality == 'av':
                    idx_img = dst_train[idx_shuffle]['frame'].to(args.device)

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
           aud_syn = torch.randn(size=(num_classes*args.ipc, channel[0], 1, im_size[0][0], im_size[0][1]), dtype=torch.float, requires_grad=True, device=args.device)
        
        if args.input_modality == 'v' or args.input_modality == 'av':
            image_syn = torch.randn(size=(num_classes*args.ipc, channel[1], im_size[1][0], im_size[1][1]), dtype=torch.float, requires_grad=True, device=args.device)
        
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            if not args.idm_aug:
                aud_real_init, img_real_init = get_aud_images_init(c, args.ipc)
                
                if args.input_modality == 'a' or args.input_modality == 'av':
                    aud_real_init = aud_real_init.detach().data
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc] = aud_real_init

                if args.input_modality == 'v' or args.input_modality == 'av':
                    img_real_init = img_real_init.detach().data
                    image_syn.data[c*args.ipc:(c+1)*args.ipc] = img_real_init
            else:
                for c in range(num_classes):
                    a_half_h, a_half_w = im_size[0][0]//2, im_size[0][1]//2
                    v_half_size = im_size[1][0]//2
                    auds_real, imgs_real = get_aud_images_init(c, args.ipc*args.idm_aug_count*args.idm_aug_count)
                    
                    start,end = 0, args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :, :a_half_h, :a_half_w] = downscale(auds_real[start:end].squeeze(2), 0.5).data.unsqueeze(2); start, end = end, end+args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :, a_half_h:, :a_half_w] = downscale(auds_real[start:end].squeeze(2), 0.5).data.unsqueeze(2); start, end = end, end+args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :, :a_half_h, a_half_w:] = downscale(auds_real[start:end].squeeze(2), 0.5).data.unsqueeze(2); start, end = end, end+args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :, a_half_h:, a_half_w:] = downscale(auds_real[start:end].squeeze(2), 0.5).data.unsqueeze(2); start, end = end, end+args.ipc

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
            # AVE数据集的herding文件路径应该由这个函数自动处理
            args.herd_path = get_herd_path(args.dataset) 
            print("Selection method: Herding")
        else:
            args.init_herding = False
            print("Selection method: Random")

        ''' 步骤 1: 使用指定方法选择并准备数据子集 (Coreset) '''
        print(f'Starting coreset selection with method: {args.selection_method}')
        
        # 将 requires_grad 设置为 False，因为我们不优化这些数据
        if image_syn is not None: image_syn.requires_grad_(False)
        if aud_syn is not None: aud_syn.requires_grad_(False)

        # 这里的填充逻辑与原始代码相同，get_aud_images_init 会根据 args.init_herding 的值自动选择方法
        # 注意：这部分填充逻辑已经包含在了 `main` 函数的 `for c in range(num_classes):` 循环中，
        # 在这个位置我们不需要再次执行，只需确保上面的 `image_syn` 和 `aud_syn` 的 requires_grad 被设为 False 即可。
        # 原始代码在 `for it...` 循环之前已经完成了初始化填充。

        print("Coreset selection finished (used initial data).")


        ''' 步骤 2: 直接评估选出的 Coreset '''
        print("Starting evaluation of the selected coreset...")
        
        # 直接使用初始化时创建的 image_syn 和 aud_syn
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
    parser.add_argument('--selection_method', type=str, default='random', choices=['random', 'herding'], help='Coreset selection method')
    parser.add_argument('--lam_cgm', type=float, default=10.0, help='weight for cross-modal gap matching loss')
    parser.add_argument('--lam_icm', type=float, default=10.0, help='weight for implicit cross matching loss')
    parser.add_argument('--lr_syn_aud', type=float, default=0.2, help='learning rate for updating synthetic audio specs')
    parser.add_argument('--lr_syn_img', type=float, default=0.2, help='learning rate for updating synthetic image')
    
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


