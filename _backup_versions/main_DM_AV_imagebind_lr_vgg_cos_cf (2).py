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
from NCFM.NCFM import match_loss, CFLossFunc


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
    
    # 核心修正：如果张量是5D的 (N, C, D, H, W)，则压缩掉多余的D维度
    if tensor.dim() == 5:
        # squeeze(2) 表示移除第2个索引位置（深度D）的维度
        tensor = tensor.squeeze(2)

    # 现在张量是4D的 (N, C, H, W)，可以安全地进行双线性插值
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
                try:
                    with open(args.herd_path, 'rb') as f:
                        herd_idx_dict = pickle.load(f)
                    idx_shuffle = herd_idx_dict[c]['av'][:min(n, len(herd_idx_dict[c]['av']))] 
                    idx_aud = dst_train_center[idx_shuffle]['audio'].to(args.device)
                    idx_img = dst_train_center[idx_shuffle]['frame'].to(args.device)
                except (FileNotFoundError, KeyError, IndexError) as e:
                    print(f"Warning: Herding failed, using random sampling: {e}")
                    idx_aud, idx_img = None, None
            else:
                idx_aud, idx_img = None, None
                try:
                    if args.init_herding:
                        with open(args.herd_path, 'rb') as f:
                            herd_idx_dict = pickle.load(f)
                        idx_shuffle = herd_idx_dict[c]['av'][:min(n, len(herd_idx_dict[c]['av']))]
                    elif len(indices_class[c]) < n:
                        idx_shuffle = np.random.permutation(indices_class[c])
                    else:
                        idx_shuffle = np.random.permutation(indices_class[c])[:min(n, len(indices_class[c]))]

                    if args.input_modality == 'a' or args.input_modality == 'av':
                        idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)

                    if args.input_modality == 'v' or args.input_modality == 'av':
                        idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
                except (IndexError, KeyError) as e:
                    print(f"Warning: Sampling failed for class {c}: {e}")
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

        def get_syn_optimizer(aud_syn, img_syn):
            param_groups = []
            if args.input_modality == 'a' or args.input_modality == 'av':
                # 将 aud_syn 包裹在列表中
                param_groups += [{'params': [aud_syn], 'lr': args.lr_syn_aud}]
            
            if args.input_modality == 'v' or args.input_modality == 'av':
                # 将 img_syn 包裹在列表中
                param_groups += [{'params': [img_syn], 'lr': args.lr_syn_img}]
            return torch.optim.SGD(param_groups, momentum=0.5)

        # ''' training '''
        optimizer_comb = get_syn_optimizer(aud_syn, image_syn)  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_comb, T_max=args.Iteration, eta_min=0.02)
        args.cf_loss_func = CFLossFunc(
            alpha_for_loss=args.alpha_for_loss, beta_for_loss=args.beta_for_loss
        )
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
                        aud_syn_eval1 = number_sign_augment(aud_syn_eval)
                    if args.input_modality == 'v' or args.input_modality == 'av':
                        image_syn_eval1 = number_sign_augment(image_syn_eval)
                    label_syn_eval = label_syn_eval.repeat(4)

                accs = []
                for it_eval in range(args.num_eval):
                    nets, net_eval = get_network_imagebind(args)
                    net_imagebind, net_classifier = nets
                    for param in list(net_imagebind.parameters()):
                            param.requires_grad = False
                    nets = (net_imagebind, net_classifier)
                    
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

            base_seed = 178645
            seed = (base_seed + it + exp) % 100000
            set_seed(seed)

            # nets, _ = get_network(args)
            nets, _ = get_network_imagebind(args)
            net_imagebind, _ = nets
            # (net_audio, net_frame, _) = nets
            # if args.input_modality == 'a' or args.input_modality == 'av':
            #     net_audio.to(args.device)
            #     net_audio.train()
            #     for param in list(net_audio.parameters()):
            #         param.requires_grad = False
            #     audio_embd = net_audio.module.embed if torch.cuda.device_count() > 1 else net_audio.embed 

            # if args.input_modality == 'v' or args.input_modality == 'av': 
            #     net_frame.to(args.device)   
            #     net_frame.train()        
            #     for param in list(net_frame.parameters()):
            #         param.requires_grad = False
            #     image_embd = net_frame.module.embed if torch.cuda.device_count() > 1 else net_frame.embed 
            net_imagebind.to(args.device)
            net_imagebind.train()
            for param in list(net_imagebind.parameters()):
                param.requires_grad = False
            imagebind_embd = net_imagebind.module.embed if torch.cuda.device_count() > 1 else net_imagebind.embed
            ''' Train synthetic data '''
            loss_avg = 0

            for c in range(num_classes):
                loss_c = torch.tensor(0.0).to(args.device)
                aud_real, img_real = get_aud_images(c, args.batch_real)

                if args.input_modality == 'a' or args.input_modality == 'av':
                    aud_real = aud_real.to(args.device)
                if args.input_modality == 'v' or args.input_modality == 'av':
                    img_real = img_real.to(args.device)

                # [修正] 直接切片，移除多余的 .reshape()
                if args.input_modality == 'a' or args.input_modality == 'av':
                    curr_aud_syn = aud_syn[c*args.ipc:(c+1)*args.ipc]
                if args.input_modality == 'v' or args.input_modality == 'av':
                    curr_img_syn = image_syn[c*args.ipc:(c+1)*args.ipc]

                if args.idm_aug:
                    if args.input_modality == 'a' or args.input_modality == 'av':
                        curr_aud_syn = number_sign_augment(curr_aud_syn)
                    if args.input_modality == 'v' or args.input_modality == 'av':
                        curr_img_syn = number_sign_augment(curr_img_syn)

                if args.dsa:
                    if args.input_modality == 'a' or args.input_modality == 'av':
                        aud_real = DiffAugment(aud_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        curr_aud_syn = DiffAugment(curr_aud_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    if args.input_modality == 'v' or args.input_modality == 'av':
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        curr_img_syn = DiffAugment(curr_img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                # Ensure inputs match ImageBind expected dimensions
                img_real = resize_to_square(img_real, 224) if img_real is not None else None
                aud_real = resize_audio_for_imagebind(aud_real) if aud_real is not None else None

                inputs_real = {}
                if img_real is not None:
                    inputs_real[ModalityType.VISION] = img_real
                if aud_real is not None:
                    inputs_real[ModalityType.AUDIO] = aud_real
                
                embeddings_real = imagebind_embd(inputs_real)
                embd_img_real = embeddings_real.get(ModalityType.VISION, torch.zeros(1, 768).to(args.device)).detach()
                embd_aud_real = embeddings_real.get(ModalityType.AUDIO, torch.zeros(1, 768).to(args.device)).detach()

                # Ensure inputs match ImageBind expected dimensions
                curr_img_syn = resize_to_square(curr_img_syn, 224) if curr_img_syn is not None else None
                curr_aud_syn = resize_audio_for_imagebind(curr_aud_syn) if curr_aud_syn is not None else None

                inputs_syn = {}
                if curr_img_syn is not None:
                    inputs_syn[ModalityType.VISION] = curr_img_syn
                if curr_aud_syn is not None:
                    inputs_syn[ModalityType.AUDIO] = curr_aud_syn
                
                embeddings_syn = imagebind_embd(inputs_syn)
                embd_img_syn = embeddings_syn.get(ModalityType.VISION, torch.zeros(1, 768).to(args.device))
                embd_aud_syn = embeddings_syn.get(ModalityType.AUDIO, torch.zeros(1, 768).to(args.device))

                ## Embedding matching
                ## Embedding matching
                if args.input_modality == 'av':                    
                    # 1. 基础分布匹配损失 (Base DM / CF Loss)
                    # 确保合成数据与真实数据在单个模态内的特征分布相似。
                    loss_base_aud = match_loss(embd_aud_real, embd_aud_syn, args)
                    loss_base_vis = match_loss(embd_img_real, embd_img_syn, args)
                    loss_base_weighted = args.lam_base * (loss_base_aud + loss_base_vis)

                    # 2. 隐式交叉匹配损失 (ICM / Cosine Loss 1)
                    # 确保“合成音频+视频”的融合特征与“真实音频+视频”的融合特征在方向上一致。
                    embd_aud_real_norm = F.normalize(embd_aud_real, p=2, dim=1)
                    embd_img_real_norm = F.normalize(embd_img_real, p=2, dim=1)
                    embd_aud_syn_norm = F.normalize(embd_aud_syn, p=2, dim=1)
                    embd_img_syn_norm = F.normalize(embd_img_syn, p=2, dim=1)
                    real_combined = embd_aud_real_norm * embd_img_real_norm
                    syn_combined = embd_aud_syn_norm * embd_img_syn_norm
                    cos_sim = torch.mm(real_combined, syn_combined.T)
                    loss_icm_weighted = args.lam_icm * torch.mean(1 - cos_sim)

                    # 3. 跨模态间隙匹配损失 (CGM / Cosine Loss 2)
                    # 确保真实与合成数据在不同模态间的相对关系（或“距离”）是一致的。
                    aud_real_avg = torch.mean(embd_aud_real, dim=0, keepdim=True)
                    img_syn_avg = torch.mean(embd_img_syn, dim=0, keepdim=True)
                    img_real_avg = torch.mean(embd_img_real, dim=0, keepdim=True)
                    aud_syn_avg = torch.mean(embd_aud_syn, dim=0, keepdim=True)
                    cross_cos_lsss = torch.mm(aud_real_avg * img_syn_avg, (img_real_avg * aud_syn_avg).T)
                    loss_cgm_weighted = args.lam_cgm * torch.mean(1 - cross_cos_lsss)

                    # 将所有加权后的损失相加
                    loss_c += loss_base_weighted + loss_icm_weighted + loss_cgm_weighted

                elif args.input_modality == 'a':
                    # 对于单模态，使用基础的分布匹配损失
                    loss_c += match_loss(embd_aud_real, embd_aud_syn, args)
                
                elif args.input_modality == 'v':
                    # 对于单模态，使用基础的分布匹配损失
                    loss_c += match_loss(embd_img_real, embd_img_syn, args)

                optimizer_comb.zero_grad()
                loss_c.backward()
                optimizer_comb.step()

                # [修正] 正确地累加每个class的loss浮点值
                loss_avg += loss_c.item()

            # [修正] 在所有类别循环结束后，计算本次迭代的平均损失
            loss_avg /= num_classes

            # [修正] 将 scheduler.step() 移到迭代末尾，确保每次迭代只执行一次
            scheduler.step()

            if it%10 == 0:
                print(f'{get_time()} iter = {it:05d}, loss = {loss_avg:.4f}, lr = {optimizer_comb.param_groups[0]["lr"]:.6f}')
                wandb.log({'train_loss': loss_avg}, step=it)

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

    parser.add_argument('--lam_cgm', type=float, default=10.0, help='weight for cross-modal gap matching loss')
    parser.add_argument('--lam_icm', type=float, default=10.0, help='weight for implicit cross matching loss')
    parser.add_argument('--lr_syn_aud', type=float, default=0.5, help='learning rate for updating synthetic audio specs')
    parser.add_argument('--lr_syn_img', type=float, default=0.5, help='learning rate for updating synthetic image')
    parser.add_argument('--num_freqs', type=int, default=4090, help='Number of frequencies for CF loss')
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
    parser.add_argument('--lam_base', type=float, default=1.0, help='weight for base distribution matching loss (CF Loss)')
    parser.add_argument('--alpha_for_loss', default=0.5, type=float)
    parser.add_argument('--beta_for_loss', default=0.5, type=float)
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


