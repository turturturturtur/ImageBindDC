import os
import copy
import argparse
import torchvision
import torch
import numpy as np
import pickle
import warnings
import torch.nn.functional as F
import time
import sys
import wandb

from utils.data_utils import get_test_dataset_origin,  get_train_dataset_origin, get_herd_path, \
    ParamDiffAug, number_sign_augment, DiffAugment, get_time
from utils.train_utils_DM import evaluate_synset_av, get_network_imagebind 
from nets.imagebind.models.imagebind_model import ModalityType
warnings.filterwarnings("ignore")
torch.set_num_threads(8)

criterion = torch.nn.CrossEntropyLoss()
def match_loss(gw_syn, gw_real):
    """
    Computes the cosine distance loss between two sets of gradients.
    This function is robust to None gradients that occur in single-modality runs.
    """
    loss = 0
    for gws, gwr in zip(gw_syn, gw_real):

        if gws is not None and gwr is not None:

            loss += 1 - torch.sum(gws * gwr) / (torch.norm(gws, 2) * torch.norm(gwr, 2) + 1e-8) # 加上一个极小值防止除以零
    return loss



def gradient_matching_loss(real_data, syn_data, real_labels, syn_labels, model, criterion):
    """
    Computes the gradient matching loss by matching gradients w.r.t. model parameters.
    This aligns with the original Dataset Condensation paper.
    """
    # Forward pass to compute the output for real and synthetic data
    real_output = model(real_data)
    syn_output = model(syn_data)

    # Compute the losses for both
    real_loss = criterion(real_output, real_labels)
    syn_loss = criterion(syn_output, syn_labels)

    # Compute gradients of the loss with respect to the model's parameters
    # The create_graph=True flag is important for the backward pass of the gradient loss
    real_grads = torch.autograd.grad(real_loss, model.parameters(), create_graph=True)
    syn_grads = torch.autograd.grad(syn_loss, model.parameters(), create_graph=True)

    # Compute the gradient matching loss (sum of MSE losses for each parameter's gradient)
    grad_loss = 0
    for rg, sg in zip(real_grads, syn_grads):
        grad_loss += F.mse_loss(rg, sg)
    
    return grad_loss

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)

def downscale(image_syn, scale_factor):
    image_syn = F.upsample(image_syn, scale_factor=scale_factor, mode='bilinear')
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
    eval_it_pool = np.arange(0, args.Iteration+1, args.interval).tolist()
    channel, im_size, num_classes, mean, std, dst_train = get_train_dataset_origin(args.dataset, args)
    # Ensure correct dimensions for ImageBind (参考main_DM_AV_imagebind_lr_vgg1.py)
    im_size = list(im_size)
    im_size[1] = (224, 224)  # Image standard size for ImageBind
    im_size[0] = (128, 204)  # ImageBind audio standard size
    print('im_size[0] (audio):', im_size[0])
    print('im_size[1] (image):', im_size[1])
    _, _, _, testloader = get_test_dataset_origin(args.dataset, args)
    args.cls_num = num_classes
    
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
                idx_aud = resize_audio_for_imagebind(idx_aud)
                idx_img = dst_train_center[idx_shuffle]['frame'].to(args.device)
                idx_img = F.interpolate(idx_img, size=(224, 224), mode='bilinear', align_corners=False)
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
                    # 确保音频数据维度正确
                    idx_aud = resize_audio_for_imagebind(idx_aud)

                if args.input_modality == 'v' or args.input_modality == 'av':
                    idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
                    # 确保图像数据维度正确
                    idx_img = F.interpolate(idx_img, size=(224, 224), mode='bilinear', align_corners=False)

            return idx_aud, idx_img
        
        def get_aud_images(c, n): 
            idx_aud, idx_img = None, None
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            if args.input_modality == 'a' or args.input_modality == 'av':
                idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)
                idx_aud = resize_audio_for_imagebind(idx_aud)
            if args.input_modality == 'v' or args.input_modality == 'av':
                idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
                idx_img = F.interpolate(idx_img, size=(224, 224), mode='bilinear', align_corners=False)
            return idx_aud, idx_img

        ''' initialize the synthetic data '''
        image_syn, aud_syn = None, None
        if args.input_modality == 'a' or args.input_modality == 'av':
            # 使用ImageBind标准音频尺寸128x204
            aud_syn = torch.randn(size=(num_classes*args.ipc, 1, 128, 204), dtype=torch.float, requires_grad=True, device=args.device)
        
        if args.input_modality == 'v' or args.input_modality == 'av':
            image_syn = torch.randn(size=(num_classes*args.ipc, channel[1], 224, 224), dtype=torch.float, requires_grad=True, device=args.device)
        
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            if not args.idm_aug:
                aud_real_init, img_real_init = get_aud_images_init(c, args.ipc)
                
                # 音频初始化 - 使用ImageBind标准尺寸128x204
                if aud_real_init is not None and (args.input_modality == 'a' or args.input_modality == 'av'):
                    print(f"DEBUG: aud_real_init shape before resize: {aud_real_init.shape}")
                    
                    # 使用统一的resize函数处理音频数据
                    aud_real_init_resized = resize_audio_for_imagebind(aud_real_init)
                    
                    # 确保批量大小匹配
                    if aud_real_init_resized.shape[0] >= args.ipc:
                        aud_syn.data[c*args.ipc:(c+1)*args.ipc] = aud_real_init_resized[:args.ipc].detach().data
                    else:
                        # 如果样本不足，使用随机初始化
                        print(f"WARNING: Not enough samples ({aud_real_init_resized.shape[0]} < {args.ipc}), using random initialization")

                # 视频初始化 - 使用标准尺寸224x224
                if img_real_init is not None and (args.input_modality == 'v' or args.input_modality == 'av'):
                    img_real_init_resized = F.interpolate(img_real_init, size=(224, 224), mode='bilinear', align_corners=False)
                    image_syn.data[c*args.ipc:(c+1)*args.ipc] = img_real_init_resized.detach().data
            else:
                for c in range(num_classes):
                    a_half_h, a_half_w = im_size[0][0]//2, im_size[0][1]//2
                    v_half_size = im_size[1][0]//2
                    auds_real, imgs_real = get_aud_images_init(c, args.ipc*args.idm_aug_count*args.idm_aug_count)
                    
                    # 确保音频数据尺寸正确
                    if auds_real is not None:
                        auds_real = resize_audio_for_imagebind(auds_real)
                        # 调整到适当的尺寸用于IDM分割
                        target_size = (a_half_h*2, a_half_w*2)
                        if auds_real.shape[-2:] != target_size:
                            auds_real = F.interpolate(auds_real, size=target_size, mode='bilinear', align_corners=False)
                    
                    start,end = 0, args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :a_half_h, :a_half_w] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, a_half_h:, :a_half_w] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :a_half_h, a_half_w:] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                    aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, a_half_h:, a_half_w:] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc

                    v_half_size = im_size[1][0]//2; start,end = 0, args.ipc
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :v_half_size, :v_half_size] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, v_half_size:, :v_half_size] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :v_half_size, v_half_size:] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, v_half_size:, v_half_size:] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc

        def get_syn_optimizer(aud_syn, img_syn):
            param_groups = []
            if args.input_modality == 'a' or args.input_modality == 'av':
                param_groups += [{'params': aud_syn, 'lr': args.lr_syn_aud}]
            
            if args.input_modality == 'v' or args.input_modality == 'av':
                param_groups += [{'params': img_syn, 'lr': args.lr_syn_img}]
            return torch.optim.SGD(param_groups, momentum=0.5)

        # ''' training '''
        optimizer_comb = get_syn_optimizer(aud_syn, image_syn)  

        print('%s training begins'%get_time())
        for it in range(args.Iteration+1):

            # +++ 这是评估模块的正确实现 +++
            if it in eval_it_pool:
                ''' Evaluate synthetic data '''
                aud_syn_eval, image_syn_eval = None, None
                if 'a' in args.input_modality:
                    aud_syn_eval = copy.deepcopy(aud_syn.detach())
                if 'v' in args.input_modality:
                    image_syn_eval = copy.deepcopy(image_syn.detach())
                label_syn_eval = copy.deepcopy(label_syn.detach())
                
                aud_syn_eval1, image_syn_eval1 = aud_syn_eval, image_syn_eval

                # 1. 正确地进行数据增强（借鉴DM脚本）
                if args.idm_aug:
                    if aud_syn_eval is not None:
                        # 修正：直接处理4D音频张量，无需squeeze/unsqueeze
                        aud_syn_eval1 = number_sign_augment(aud_syn_eval)
                    if image_syn_eval is not None:
                        image_syn_eval1 = number_sign_augment(image_syn_eval)
                    label_syn_eval = label_syn_eval.repeat(4)

                accs = []
                for it_eval in range(args.num_eval):

                    #    评估函数内部会自己创建全新的、完全可训练的网络
                    nets, net_eval = get_network_imagebind(args)

                    # 【重要】我们在这里不冻结任何参数，以保证DC评估的公平性
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
                    grid = torchvision.utils.make_grid(aud_syn_vis, nrow=max(10, args.ipc), normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Audio": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                if args.input_modality == 'v' or args.input_modality == 'av':
                    image_syn_vis = copy.deepcopy(image_syn_eval.detach().cpu())
                    torch.save(image_syn_vis, args.syn_data_path + f'/exp_{exp}_imgSyn_{it}.pt')
                    for ch in range(channel[1]):
                        image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                    image_syn_vis[image_syn_vis<0] = 0.0
                    image_syn_vis[image_syn_vis>1] = 1.0
                    grid = torchvision.utils.make_grid(image_syn_vis, nrow=max(10, args.ipc), normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Image": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
            base_seed = 178645
            seed = (base_seed + it + exp) % 100000
            set_seed(seed)

            # 1. 调用新的网络构建函数，注意pretrained=False
            nets, net_eval = get_network_imagebind(args, pretrained=False)
            # 2. 解包为新的模型组件
            net_imagebind, net_classifier = nets

            # 将模型移动到设备
            net_imagebind.to(args.device)
            net_classifier.to(args.device)

            # 确保所有模型组件都处于可训练状态 (这段逻辑无需改变)
            for net_component in nets:
                if net_component is not None:
                    net_component.train() 
                    for param in net_component.parameters():
                        param.requires_grad = True

            # 3. 收集新架构下的所有可训练参数
            net_parameters = list(net_imagebind.parameters()) + list(net_classifier.parameters())
            imagebind_embd = net_imagebind.module.embed if torch.cuda.device_count() > 1 else net_imagebind.embed

            ''' Train synthetic data (DC Method) '''
            loss_avg = 0
            loss = torch.tensor(0.0).to(args.device)
    # +++ 这是修改后的新函数 +++
            def get_output(aud_data, img_data,embed_fn):
    # 1. 准备ImageBind的输入字典
                inputs = {}
                if aud_data is not None:
                    inputs[ModalityType.AUDIO] = aud_data
                if img_data is not None:
                    inputs[ModalityType.VISION] = img_data
                
                if not inputs:
                    return None

                # 2. 使用统一的ImageBind模型提取所有模态的特征
                embeddings = embed_fn(inputs) 
                feat_sound = embeddings.get(ModalityType.AUDIO)
                feat_frame = embeddings.get(ModalityType.VISION)

                # 3. 进行特征融合（采用1024维相加方案）
                if feat_sound is not None and feat_frame is not None:
                    feat = feat_sound + feat_frame
                elif feat_sound is not None:
                    feat = feat_sound
                elif feat_frame is not None:
                    feat = feat_frame
                else:
                    return None
                
                # 4. 将融合后的单一特征向量送入分类器
                return net_classifier(feat)
            
            for c in range(num_classes):
                
                # --- 您原来的代码：获取真实和合成数据的梯度 ---
                aud_real, img_real = get_aud_images(c, args.batch_real)
                real_data_count = aud_real.shape[0] if aud_real is not None else (img_real.shape[0] if img_real is not None else 0)
                if real_data_count == 0: continue
                
                lab_real = torch.tensor([c] * real_data_count, device=args.device, dtype=torch.long)
                output_real = get_output(aud_real, img_real, imagebind_embd)

                if output_real is None: continue
                
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters, allow_unused=True)
                gw_real = list((g.detach().clone() if g is not None else g for g in gw_real))


                if 'a' in args.input_modality:
                    # 直接使用正确的4D维度，无需reshape添加额外维度
                    curr_aud_syn = aud_syn[c*args.ipc:(c+1)*args.ipc]
                if 'v' in args.input_modality:
                    curr_img_syn = image_syn[c*args.ipc:(c+1)*args.ipc]
                output_syn = get_output(curr_aud_syn, curr_img_syn,imagebind_embd)


                if output_syn is None: continue

                lab_syn = torch.tensor([c] * args.ipc, device=args.device, dtype=torch.long)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True, allow_unused=True)
                # --- 逻辑到此保持不变 ---

                # [核心修改] 只累加损失张量，为后续统一的backward()做准备
                loss += match_loss(gw_syn, gw_real)


            # 2. 在所有类别的损失都累加完成后，统一进行一次反向传播和优化
            optimizer_comb.zero_grad()
            loss.backward()
            optimizer_comb.step()

            # 3. 在优化步骤完成后，进行日志记录（每10次迭代）
            #    这里的 loss.item() 已经是当前迭代的总损失了
            if it%10 == 0:
                # 确保 num_classes 不是0，防止除零错误
                avg_loss_this_iter = loss.item() / num_classes if num_classes > 0 else 0
                print(f'{get_time()} iter = {it:05d}, loss = {avg_loss_this_iter:.4f}, lr = {optimizer_comb.param_groups[0]["lr"]:.6f}')
                wandb.log({'train_loss': avg_loss_this_iter}, step=it)

        print('experiment run save')
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
    parser.add_argument('--batch_real', type=int, default=32, help='batch size for real data')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')

    parser.add_argument('--base_syn_data_dir', type=str, default='data/syn_data_train', help='a/v/av')

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

