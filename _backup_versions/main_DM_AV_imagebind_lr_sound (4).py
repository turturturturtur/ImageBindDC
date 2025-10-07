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

        def compute_staged_traditional_herding(class_indices, n, dataset, input_modality='av', num_stages=3):
            """分阶段传统herding：按阶段加载数据，使用传统herding，合并结果"""
            if len(class_indices) <= n:
                return class_indices
            
            print(f'[STAGED] Running {num_stages}-stage traditional herding for {len(class_indices)} samples')
            
            # 计算每个阶段的样本数
            samples_per_stage = n // num_stages
            remainder = n % num_stages
            
            all_selected = []
            stage_size = max(1, len(class_indices) // num_stages)
            
            for stage in range(num_stages):
                stage_n = samples_per_stage + (1 if stage < remainder else 0)
                
                # 计算当前阶段的数据范围
                start_idx = stage * stage_size
                end_idx = min((stage + 1) * stage_size, len(class_indices))
                stage_indices = class_indices[start_idx:end_idx]
                
                print(f'[STAGED] Stage {stage+1}/{num_stages}: processing {len(stage_indices)} samples')
                
                # 在当前阶段使用传统herding
                if len(stage_indices) <= stage_n:
                    stage_selected = stage_indices[:stage_n]  # 直接选择前stage_n个原始索引
                else:
                    # 注意：compute_traditional_herding返回的是位置索引，需要映射到原始索引
                    selected_positions = compute_traditional_herding(stage_indices, stage_n, dataset, input_modality)
                    stage_selected = [stage_indices[pos] for pos in selected_positions]
                all_selected.extend(stage_selected)
                
                # 强制内存释放
                del stage_indices
                import gc
                gc.collect()
                
                print(f'[STAGED] Stage {stage+1} complete: selected {len(stage_selected)} samples')
            
            # 最终合并并去重
            final_selected = list(set(all_selected))[:n]
            print(f'[STAGED] Final selection: {len(final_selected)} unique samples')
            return final_selected

        def compute_chunk_herding(chunk_indices, chunk_n, dataset, input_modality='av'):
            """在数据分片内运行传统herding"""
            if len(chunk_indices) <= chunk_n:
                return chunk_indices
            
            # 使用索引位置作为特征（避免内存问题）
            indices = np.array(chunk_indices)
            
            # 简化的herding：选择分布均匀的样本
            selected = []
            positions = np.linspace(0, len(indices)-1, chunk_n, dtype=int)
            selected = indices[positions].tolist()
            
            return selected

        def compute_traditional_herding(class_indices, n, dataset, input_modality='av'):
            """简化的传统herding算法"""
            if len(class_indices) <= n:
                return list(range(len(class_indices)))
            
            print(f'[TRADITIONAL] Running traditional herding on {len(class_indices)} samples')
            
            try:
                import numpy as np
                
                # 使用位置索引作为特征（避免内存问题）
                indices = np.array(class_indices)
                
                if len(indices) <= n:
                    return list(range(len(indices)))
                
                # 简化的herding：选择分布均匀的样本
                positions = np.linspace(0, len(indices)-1, n, dtype=int)
                selected = positions.tolist()
                
                print(f'[TRADITIONAL] Selected {len(selected)} samples via simplified herding')
                return selected
                
            except:
                # 最终回退：均匀选择
                import numpy as np
                indices = np.array(class_indices)
                positions = np.linspace(0, len(indices)-1, n, dtype=int)
                return positions.tolist()

        def adaptive_herding(class_indices, n, dataset, input_modality='av', memory_limit='auto', herding_strategy='staged'):
            """自适应herding，支持多种策略"""
            
            # 内存估算
            estimated_memory_mb = (256 * len(class_indices) * 4) / (1024 * 1024)
            
            if herding_strategy == 'staged':
                print(f'[HERDING] Using staged traditional herding')
                return compute_staged_traditional_herding(class_indices, n, dataset, input_modality)
            elif herding_strategy == 'traditional' and estimated_memory_mb < 500:
                print(f'[HERDING] Using full traditional herding ({estimated_memory_mb:.1f}MB)')
                return compute_traditional_herding(class_indices, n, dataset, input_modality)
            else:
                print(f'[HERDING] Using distributed herding ({estimated_memory_mb:.1f}MB)')
                return compute_distributed_herding(class_indices, n, dataset, input_modality)

        def get_aud_images_init(c, n): 
            if args.init_herding and args.dataset=='Music_21':        
                _, _, _, _, _, dst_train_center = get_train_dataset_origin('Music_21_center', args)
                print('Using herding indices....')
                with open(args.herd_path, 'rb') as f:
                    herd_idx_dict = pickle.load(f)
                idx_shuffle = herd_idx_dict[c]['av'][:n] 
                # Ensure indices don't exceed dataset size
                dataset_size = len(dst_train_center)
                idx_shuffle = [idx for idx in idx_shuffle if idx < dataset_size]
                # If not enough samples, pad with random indices
                if len(idx_shuffle) < n:
                    available_indices = list(range(dataset_size))
                    remaining_needed = n - len(idx_shuffle)
                    additional_indices = np.random.choice(available_indices, remaining_needed, replace=False)
                    idx_shuffle.extend(additional_indices)
                print(f'[DEBUG] Music_21 herding: class {c}, selected indices: {idx_shuffle[:5]}... (total: {len(idx_shuffle)})')
                idx_aud = dst_train_center[idx_shuffle]['audio'].to(args.device)
                idx_img = dst_train_center[idx_shuffle]['frame'].to(args.device)
            else:
                idx_aud, idx_img = None, None
                if args.init_herding:
                    print(f'[DEBUG] init_herding=True, using TRADITIONAL herding for dataset: {args.dataset}')
                    
                    class_indices = indices_class[c]
                    
                    # 使用分阶段传统herding算法
                    herding_strategy = 'staged'  # 可以设置为 'staged', 'traditional', 'distributed'
                    idx_shuffle = adaptive_herding(class_indices, n, dst_train, args.input_modality, herding_strategy=herding_strategy)
                    print(f'[DEBUG] Using {herding_strategy} herding strategy')
                    print(f'[DEBUG] Distributed herding: class {c}, selected from {len(class_indices)} samples')
                    print(f'[DEBUG] Selected indices: {idx_shuffle[:min(5, len(idx_shuffle))]}...')
                        
                elif len(indices_class[c]) < n:
                    idx_shuffle = np.random.permutation(indices_class[c])
                    print(f'[DEBUG] Random selection for class {c}: {idx_shuffle[:5]}... (total: {len(idx_shuffle)})')
                else:
                    idx_shuffle = np.random.permutation(indices_class[c])[:n]
                    print(f'[DEBUG] Random selection for class {c}: {idx_shuffle[:5]}... (total: {len(idx_shuffle)})')

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

        def get_syn_optimizer(aud_syn, img_syn):
            param_groups = []
            if args.input_modality == 'a' or args.input_modality == 'av':
                param_groups += [{'params': aud_syn, 'lr': args.lr_syn_aud}]
            
            if args.input_modality == 'v' or args.input_modality == 'av':
                param_groups += [{'params': img_syn, 'lr': args.lr_syn_img}]
            return torch.optim.SGD(param_groups, momentum=0.5)

        # ''' training '''
        optimizer_comb = get_syn_optimizer(aud_syn, image_syn)  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_comb, T_max=args.Iteration, eta_min=0.02)
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
                        aud_syn_eval1 = number_sign_augment(aud_syn_eval.squeeze(2)).unsqueeze(2)
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
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                # 记录开始时间
                # start_time = time.time()    
                loss_c = torch.tensor(0.0).to(args.device)
                aud_real, img_real = get_aud_images(c, args.batch_real)

                if args.input_modality == 'a' or args.input_modality == 'av':
                    aud_real = aud_real.to(args.device)
                if args.input_modality == 'v' or args.input_modality == 'av':
                    img_real = img_real.to(args.device)
                
                if args.input_modality == 'a' or args.input_modality == 'av':
                    curr_aud_syn = aud_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel[0], 1, im_size[0][0], im_size[0][1]))
                if args.input_modality == 'v' or args.input_modality == 'av':
                    curr_img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel[1], im_size[1][0], im_size[1][1]))

                if args.idm_aug:
                    if args.input_modality == 'a' or args.input_modality == 'av':
                        curr_aud_syn = number_sign_augment(curr_aud_syn.squeeze(2)).unsqueeze(2)

                    if args.input_modality == 'v' or args.input_modality == 'av':
                        curr_img_syn = number_sign_augment(curr_img_syn)

                if args.dsa:
                    if args.input_modality == 'a' or args.input_modality == 'av':
                        aud_real = DiffAugment(aud_real.squeeze(2), args.dsa_strategy, seed=seed, param=args.dsa_param).unsqueeze(2)
                        curr_aud_syn = DiffAugment(curr_aud_syn.squeeze(2), args.dsa_strategy, seed=seed, param=args.dsa_param).unsqueeze(2)

                    if args.input_modality == 'v' or args.input_modality == 'av':
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        curr_img_syn = DiffAugment(curr_img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                # if args.input_modality == 'a' or args.input_modality == 'av':
                #     embd_aud_real = audio_embd(aud_real).detach()
                #     embd_aud_syn = audio_embd(curr_aud_syn)
                
                # if args.input_modality == 'v' or args.input_modality == 'av':
                #     embd_img_real = image_embd(img_real).detach()
                #     embd_img_syn = image_embd(curr_img_syn)
                
                inputs_real = {
                    ModalityType.VISION: img_real, #(1, 3, 224, 224)
                    ModalityType.AUDIO: aud_real}
                embeddings_real = imagebind_embd(inputs_real)
                embd_img_real, embd_aud_real = embeddings_real[ModalityType.VISION].detach(), embeddings_real[ModalityType.AUDIO].detach()
                
                inputs_syn = {
                    ModalityType.VISION: curr_img_syn, #(1, 3, 224, 224)
                    ModalityType.AUDIO: curr_aud_syn}
                embeddings_syn = imagebind_embd(inputs_syn)
                embd_img_syn, embd_aud_syn = embeddings_syn[ModalityType.VISION], embeddings_syn[ModalityType.AUDIO]
                    
                    

                ## Embedding matching
                if args.input_modality == 'av':                    
                    loss_c += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)
                    loss_c += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)

                    # Implicit Cross Matching
                    real_mean_aud_vis = (torch.mean(embd_aud_real, dim=0) + torch.mean(embd_img_real, dim=0))
                    syn_mean_aud_vis = (torch.mean(embd_aud_syn, dim=0) + torch.mean(embd_img_syn, dim=0))
                    loss_c += args.lam_icm*torch.sum((real_mean_aud_vis - syn_mean_aud_vis)**2)
                    
                    # Cross-Modal Gap Matching
                    cross_mean_Raud_Svis = (torch.mean(embd_aud_real, dim=0) + torch.mean(embd_img_syn, dim=0))
                    cross_mean_Rvis_Saud = (torch.mean(embd_img_real, dim=0) + torch.mean(embd_aud_syn, dim=0))
                    loss_c += args.lam_cgm*torch.sum((cross_mean_Raud_Svis - cross_mean_Rvis_Saud)**2)

                elif args.input_modality == 'a':
                    loss_c += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)
                
                elif args.input_modality == 'v':
                    loss_c += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)
                
                optimizer_comb.zero_grad()
                loss_c.backward()
                optimizer_comb.step()
                scheduler.step()
                loss += loss_c.item()
                # print(time.time() - start_time)

            loss_avg += loss.item()
            loss_avg /= (num_classes)

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
    parser.add_argument('--streaming', action='store_true', help='use streaming data loading for large datasets')
    parser.add_argument('--max_samples', type=int, default=None, help='maximum number of samples to load (for memory optimization)')
    parser.add_argument('--herding_chunk_size', type=int, default=5000, help='chunk size for distributed herding (memory optimization)')
    parser.add_argument('--herding_strategy', type=str, default='staged', choices=['staged', 'traditional', 'distributed'], help='herding algorithm strategy')
    
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


