# import os
# import copy
# import argparse
# import torchvision
# import wandb
# import torch
# import numpy as np
# import pickle
# import warnings
# import torch.nn.functional as F
# import time

# from utils.data_utils import get_test_dataset, get_train_dataset, get_herd_path, \
#     ParamDiffAug, number_sign_augment, DiffAugment, get_time
# from utils.train_utils_DM import evaluate_synset_av, get_network

# warnings.filterwarnings("ignore")
# torch.set_num_threads(8)

# def set_seed(seed_value):
#     torch.manual_seed(seed_value)
#     torch.cuda.manual_seed_all(seed_value)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed_value)

# def downscale(image_syn, scale_factor):
#     image_syn = F.upsample(image_syn, scale_factor=scale_factor, mode='bilinear')
#     return image_syn

# def main(args):
#     eval_it_pool = np.arange(0, args.Iteration+1, args.interval).tolist()
#     channel, im_size, num_classes, mean, std, dst_train = get_train_dataset(args.dataset, args)
#     _, _, _, testloader = get_test_dataset(args.dataset, args)
#     args.cls_num = num_classes
    
#     accs_all_exps = []

#     for exp in range(args.num_exp):
#         print('\n================== Exp %d ==================\n '%exp)
        
#         if args.wandb_disable:
#             wandb.init(mode="disabled")
#         else:
#             wandb.init(sync_tensorboard=False,
#                     project="AVDD",
#                     config=args,
#                     name = f'{args.id}_exp-{exp}')

#         base_seed = 178645
#         seed = (base_seed + exp) % 100000
#         set_seed(seed)

#         labels_all = []
#         indices_class = [[] for c in range(num_classes)]

#         labels_all = [dst_train[i]['label'] for i in range(len(dst_train))]
#         for i, lab in enumerate(labels_all):
#             indices_class[lab].append(i)
#         labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

#         for c in range(num_classes):
#             print('class c = %d: %d real images'%(c, len(indices_class[c])))

#         if args.init_herding:
#             args.herd_path = get_herd_path(args.dataset)

#         def get_aud_images_init(c, n): 
#             if args.init_herding and args.dataset=='Music_21':        
#                 _, _, _, _, _, dst_train_center = get_train_dataset('Music_21_center', args)
#                 print('Using herding indices....')
#                 with open(args.herd_path, 'rb') as f:
#                     herd_idx_dict = pickle.load(f)
#                 idx_shuffle = herd_idx_dict[c]['av'][:n] 
#                 idx_aud = dst_train_center[idx_shuffle]['audio'].to(args.device)
#                 idx_img = dst_train_center[idx_shuffle]['frame'].to(args.device)
#             else:
#                 idx_aud, idx_img = None, None
#                 if args.init_herding:
#                     with open(args.herd_path, 'rb') as f:
#                         herd_idx_dict = pickle.load(f)
#                     idx_shuffle = herd_idx_dict[c]['av'][:n]
#                 elif len(indices_class[c]) < n:
#                     idx_shuffle = np.random.permutation(indices_class[c])
#                 else:
#                     idx_shuffle = np.random.permutation(indices_class[c])[:n]

#                 if args.input_modality == 'a' or args.input_modality == 'av':
#                     idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)

#                 if args.input_modality == 'v' or args.input_modality == 'av':
#                     idx_img = dst_train[idx_shuffle]['frame'].to(args.device)

#             return idx_aud, idx_img
        
#         def get_aud_images(c, n): 
#             idx_aud, idx_img = None, None
#             idx_shuffle = np.random.permutation(indices_class[c])[:n].tolist()
#             if args.input_modality == 'a' or args.input_modality == 'av':
#                 idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)
#             if args.input_modality == 'v' or args.input_modality == 'av':
#                 idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
#             return idx_aud, idx_img

#         ''' initialize the synthetic data '''
#         image_syn, aud_syn = None, None
#         if args.input_modality == 'a' or args.input_modality == 'av':
#             aud_syn = torch.randn(size=(num_classes*args.ipc, channel[0], 1, im_size[0][0], im_size[0][1]), dtype=torch.float, requires_grad=True, device=args.device)
        
#         if args.input_modality == 'v' or args.input_modality == 'av':
#             image_syn = torch.randn(size=(num_classes*args.ipc, channel[1], im_size[1][0], im_size[1][1]), dtype=torch.float, requires_grad=True, device=args.device)
        
#         label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

#         print('initialize synthetic data from random real images')
#         for c in range(num_classes):
#             if not args.idm_aug:
#                 aud_real_init, img_real_init = get_aud_images_init(c, args.ipc)
                
#                 if args.input_modality == 'a' or args.input_modality == 'av':
#                     aud_real_init = aud_real_init.detach().data
#                     aud_syn.data[c*args.ipc:(c+1)*args.ipc] = aud_real_init

#                 if args.input_modality == 'v' or args.input_modality == 'av':
#                     img_real_init = img_real_init.detach().data
#                     image_syn.data[c*args.ipc:(c+1)*args.ipc] = img_real_init
#             else:
#                 for c in range(num_classes):
#                     a_half_h, a_half_w = im_size[0][0]//2, im_size[0][1]//2
#                     v_half_size = im_size[1][0]//2
#                     auds_real, imgs_real = get_aud_images_init(c, args.ipc*args.idm_aug_count*args.idm_aug_count)
                    
#                     start,end = 0, args.ipc
#                     aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :a_half_h, :a_half_w] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
#                     aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, a_half_h:, :a_half_w] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
#                     aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :a_half_h, a_half_w:] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
#                     aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, a_half_h:, a_half_w:] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc

#                     v_half_size = im_size[1][0]//2; start,end = 0, args.ipc
#                     image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :v_half_size, :v_half_size] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
#                     image_syn.data[c*args.ipc:(c+1)*args.ipc, :, v_half_size:, :v_half_size] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
#                     image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :v_half_size, v_half_size:] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
#                     image_syn.data[c*args.ipc:(c+1)*args.ipc, :, v_half_size:, v_half_size:] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc

#         def get_syn_optimizer(aud_syn, img_syn):
#             param_groups = []
#             if args.input_modality == 'a' or args.input_modality == 'av':
#                 param_groups += [{'params': aud_syn, 'lr': args.lr_syn_aud}]
            
#             if args.input_modality == 'v' or args.input_modality == 'av':
#                 param_groups += [{'params': img_syn, 'lr': args.lr_syn_img}]
#             return torch.optim.SGD(param_groups, momentum=0.5)

#         # ''' training '''
#         optimizer_comb = get_syn_optimizer(aud_syn, image_syn)  

#         print('%s training begins'%get_time())
#         for it in range(args.Iteration+1):

#             if it in eval_it_pool:
#                 ''' Evaluate synthetic data '''
#                 aud_syn_eval, image_syn_eval = None, None
#                 if args.input_modality == 'a' or args.input_modality == 'av':
#                     aud_syn_eval = copy.deepcopy(aud_syn.detach())
#                 if args.input_modality == 'v' or args.input_modality == 'av':
#                     image_syn_eval = copy.deepcopy(image_syn.detach())
#                 label_syn_eval = copy.deepcopy(label_syn.detach())
                
#                 # if args.idm_aug:
#                 #     if args.input_modality == 'a' or args.input_modality == 'av':
#                 #         aud_syn_eval1 = number_sign_augment(aud_syn_eval)
#                 #     if args.input_modality == 'v' or args.input_modality == 'av':
#                 #         image_syn_eval1 = number_sign_augment(image_syn_eval)
#                 #     label_syn_eval = label_syn_eval.repeat(4)
#                 # 添加变量初始化
#                 aud_syn_eval1, image_syn_eval1 = aud_syn_eval, image_syn_eval
#                 if args.idm_aug:
#                     if args.input_modality == 'a' or args.input_modality == 'av':
#                         aud_syn_eval1 = number_sign_augment(aud_syn_eval.squeeze(2)).unsqueeze(2)
#                     if args.input_modality == 'v' or args.input_modality == 'av':
#                         image_syn_eval1 = number_sign_augment(image_syn_eval)
#                     label_syn_eval = label_syn_eval.repeat(4)

#                 accs = []
#                 for it_eval in range(args.num_eval):
#                     nets, net_eval = get_network(args)
#                     acc = evaluate_synset_av(nets, net_eval, aud_syn_eval1, image_syn_eval1, label_syn_eval, testloader, args)
#                     accs.append(acc)
#                     print(f'it_eval: {it_eval} Val acc: {acc:.2f}%')

#                 wandb.log({'eval_acc': np.mean(accs)}, step=it)
#                 print(f'Mean eval at it: {it} Val acc: {np.mean(accs):.2f}%')

#                 if it == args.Iteration: # record the final results
#                     accs_all_exps += accs
                
#                 ''' visualize and save '''
#                 if args.input_modality == 'a' or args.input_modality == 'av':
#                     aud_syn_vis = copy.deepcopy(aud_syn_eval.detach().cpu())
#                     torch.save(aud_syn_vis, args.syn_data_path + f'/exp_{exp}_audSyn_{it}.pt')
#                     grid = torchvision.utils.make_grid(aud_syn_vis, nrow=max(10, args.ipc), normalize=True, scale_each=True)
#                     # wandb.log({"Synthetic_Audio": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

#                 if args.input_modality == 'v' or args.input_modality == 'av':
#                     image_syn_vis = copy.deepcopy(image_syn_eval.detach().cpu())
#                     torch.save(image_syn_vis, args.syn_data_path + f'/exp_{exp}_imgSyn_{it}.pt')
#                     for ch in range(channel[1]):
#                         image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
#                     image_syn_vis[image_syn_vis<0] = 0.0
#                     image_syn_vis[image_syn_vis>1] = 1.0
#                     grid = torchvision.utils.make_grid(image_syn_vis, nrow=max(10, args.ipc), normalize=True, scale_each=True)
#                     # wandb.log({"Synthetic_Image": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

#             base_seed = 178645
#             seed = (base_seed + it + exp) % 100000
#             set_seed(seed)

#             nets, _ = get_network(args)
#             (net_audio, net_frame, _) = nets
#             if args.input_modality == 'a' or args.input_modality == 'av':
#                 net_audio.to(args.device)
#                 net_audio.train()
#                 for param in list(net_audio.parameters()):
#                     param.requires_grad = False
#                 audio_embd = net_audio.module.embed if torch.cuda.device_count() > 1 else net_audio.embed 

#             if args.input_modality == 'v' or args.input_modality == 'av': 
#                 net_frame.to(args.device)   
#                 net_frame.train()        
#                 for param in list(net_frame.parameters()):
#                     param.requires_grad = False
#                 image_embd = net_frame.module.embed if torch.cuda.device_count() > 1 else net_frame.embed 

#             ''' Train synthetic data '''
#             loss_avg = 0
#             loss = torch.tensor(0.0).to(args.device)
#             for c in range(num_classes):
#                 loss_c = torch.tensor(0.0).to(args.device)
#                 aud_real, img_real = get_aud_images(c, args.batch_real)

#                 if args.input_modality == 'a' or args.input_modality == 'av':
#                     aud_real = aud_real.to(args.device)
#                 if args.input_modality == 'v' or args.input_modality == 'av':
#                     img_real = img_real.to(args.device)
                
#                 if args.input_modality == 'a' or args.input_modality == 'av':
#                     curr_aud_syn = aud_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel[0], 1, im_size[0][0], im_size[0][1]))
#                 if args.input_modality == 'v' or args.input_modality == 'av':
#                     curr_img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel[1], im_size[1][0], im_size[1][1]))

#                 if args.idm_aug:
#                     if args.input_modality == 'a' or args.input_modality == 'av':
#                         curr_aud_syn = number_sign_augment(curr_aud_syn)

#                     if args.input_modality == 'v' or args.input_modality == 'av':
#                         curr_img_syn = number_sign_augment(curr_img_syn)

#                 if args.dsa:
#                     if args.input_modality == 'a' or args.input_modality == 'av':
#                         aud_real = DiffAugment(aud_real.squeeze(2), args.dsa_strategy, seed=seed, param=args.dsa_param).unsqueeze(2)
#                         curr_aud_syn = DiffAugment(curr_aud_syn.squeeze(2), args.dsa_strategy, seed=seed, param=args.dsa_param).unsqueeze(2)

#                     if args.input_modality == 'v' or args.input_modality == 'av':
#                         img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
#                         curr_img_syn = DiffAugment(curr_img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

#                 if args.input_modality == 'a' or args.input_modality == 'av':
#                     embd_aud_real = audio_embd(aud_real).detach()
#                     embd_aud_syn = audio_embd(curr_aud_syn)
                
#                 if args.input_modality == 'v' or args.input_modality == 'av':
#                     embd_img_real = image_embd(img_real).detach()
#                     embd_img_syn = image_embd(curr_img_syn)

#                 ## Embedding matching
#                 if args.input_modality == 'av':                    
#                     loss_c += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)
#                     loss_c += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)

#                     # Implicit Cross Matching
#                     real_mean_aud_vis = (torch.mean(embd_aud_real, dim=0) + torch.mean(embd_img_real, dim=0))
#                     syn_mean_aud_vis = (torch.mean(embd_aud_syn, dim=0) + torch.mean(embd_img_syn, dim=0))
#                     loss_c += args.lam_icm*torch.sum((real_mean_aud_vis - syn_mean_aud_vis)**2)
                    
#                     # Cross-Modal Gap Matching
#                     cross_mean_Raud_Svis = (torch.mean(embd_aud_real, dim=0) + torch.mean(embd_img_syn, dim=0))
#                     cross_mean_Rvis_Saud = (torch.mean(embd_img_real, dim=0) + torch.mean(embd_aud_syn, dim=0))
#                     loss_c += args.lam_cgm*torch.sum((cross_mean_Raud_Svis - cross_mean_Rvis_Saud)**2)

#                 elif args.input_modality == 'a':
#                     loss_c += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)
                
#                 elif args.input_modality == 'v':
#                     loss_c += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)
                
#                 optimizer_comb.zero_grad()
#                 loss_c.backward()
#                 optimizer_comb.step()

#                 loss += loss_c.item()

#             loss_avg += loss.item()
#             loss_avg /= (num_classes)

#             if it%10 == 0:
#                 print(f'{get_time()} iter = {it:05d}, loss = {loss_avg:.4f}, lr = {optimizer_comb.param_groups[0]["lr"]:.6f}')
#                 wandb.log({'train_loss': loss_avg}, step=it)

#         print('experiment run save')
#         wandb.finish()

#     print('\n==================== Final Results ====================\n')
#     accs = accs_all_exps
#     print('Run %d experiments, random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, len(accs), np.mean(accs), np.std(accs)))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Parameter Processing')
#     #training parameters
#     parser.add_argument('--dataset', type=str, default='VGG_subset', help='dataset')
#     parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
#     parser.add_argument('--Iteration', type=int, default=5000, help='training iterations')
#     parser.add_argument('--num_exp', type=int, default=3, help='the number of experiments')    

#     parser.add_argument('--lam_cgm', type=float, default=10.0, help='weight for cross-modal gap matching loss')
#     parser.add_argument('--lam_icm', type=float, default=10.0, help='weight for implicit cross matching loss')
#     parser.add_argument('--lr_syn_aud', type=float, default=0.2, help='learning rate for updating synthetic audio specs')
#     parser.add_argument('--lr_syn_img', type=float, default=0.2, help='learning rate for updating synthetic image')
    
#     #evaluation parameters
#     parser.add_argument('--epoch_eval_train', type=int, default=30, help='epochs to train a model with synthetic data')
#     parser.add_argument('--interval', type=int, default=1000, help='interval to evaluate the synthetic data')
#     parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')

#     parser.add_argument('--arch_sound', type=str, default='convNet', help='convNet')
#     parser.add_argument('--weights_sound', type=str, default='', help='weights for sound')   
#     parser.add_argument('--arch_frame', type=str, default='convNet', help='convNet')
#     parser.add_argument('--weights_frame', type=str, default='', help='weights for frame')
#     parser.add_argument('--arch_classifier', type=str, default='ensemble', help='ensemble')
#     parser.add_argument('--weights_classifier', type=str, default='', help='weights for classifier')

#     parser.add_argument('--lr_frame', type=float, default=1e-4, help='learning rate for updating synthetic audio specs')
#     parser.add_argument('--lr_sound', type=float, default=1e-3, help='sound learning rate')    
#     parser.add_argument('--lr_classifier', type=float, default=1e-3, help='classifier learning rate')
#     parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
#     parser.add_argument('--weight_decay', type=float, default=1e-4, help='classifier learning rate')
#     parser.add_argument('--batch_syn', type=int, default=32, help='batch size for syn data')
#     parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')

#     #data parameters
#     parser.add_argument('--init_herding', action='store_true', help='init using herding or not')
#     parser.add_argument('--input_modality', type=str, default='av', help='a/v/av')
#     parser.add_argument('--idm_aug_count', type=int, default=2, help='number of images per image during IDM')
#     parser.add_argument('--idm_aug', action='store_true', help='use Augmentation or not')
#     parser.add_argument('--wandb_disable', action='store_true', help='wandb disable')
#     parser.add_argument('--batch_real', type=int, default=128, help='batch size for real data')
#     parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')

#     parser.add_argument('--base_syn_data_dir', type=str, default='data/syn_data_train', help='a/v/av')

#     args = parser.parse_args()
#     args.dsa_param = ParamDiffAug()
#     args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
#     args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     args.lr_steps_interval = 10
#     args.lr_steps = np.arange(args.lr_steps_interval, args.epoch_eval_train, args.lr_steps_interval).tolist()

#     args.id = f'{args.dataset}_ipc-{args.ipc}'
#     args.syn_data_path = os.path.join(args.base_syn_data_dir, args.id)
#     os.makedirs(args.syn_data_path, exist_ok=True)
    
#     main(args)






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
from NCFM.NCFM import match_loss, cailb_loss, mutil_layer_match_loss, CFLossFunc
from utils.old_data_utils import get_test_dataset, get_train_dataset, get_herd_path, \
    ParamDiffAug, number_sign_augment, DiffAugment, get_time
from utils.old_train_utils_DM import evaluate_synset_av, get_network

warnings.filterwarnings("ignore")
torch.set_num_threads(8)

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)

def downscale(image_syn, scale_factor):
    image_syn =F.interpolate(image_syn, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    return image_syn

def main(args):
    eval_it_pool = np.arange(0, args.Iteration+1, args.interval).tolist()
    channel, im_size, num_classes, mean, std, dst_train = get_train_dataset(args.dataset, args)
    _, _, _, testloader = get_test_dataset(args.dataset, args)
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
                _, _, _, _, _, dst_train_center = get_train_dataset('Music_21_center', args)
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
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            if args.input_modality == 'a' or args.input_modality == 'av':
                idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)
            if args.input_modality == 'v' or args.input_modality == 'av':
                idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
            return idx_aud, idx_img

        ''' initialize the synthetic data '''
        image_syn, aud_syn = None, None
        if args.input_modality == 'a' or args.input_modality == 'av':
            aud_syn = torch.randn(size=(num_classes*args.ipc, channel[0], im_size[0][0], im_size[0][1]), dtype=torch.float, requires_grad=True, device=args.device)
        
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
                    if args.input_modality == 'a' or args.input_modality == 'av':
                        aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :a_half_h, :a_half_w] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                        aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, a_half_h:, :a_half_w] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                        aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, :a_half_h, a_half_w:] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                        aud_syn.data[c*args.ipc:(c+1)*args.ipc, :, a_half_h:, a_half_w:] = downscale(auds_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                    if args.input_modality == 'v' or args.input_modality == 'av':
                        v_half_size = im_size[1][0]//2; start,end = 0, args.ipc
                        image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :v_half_size, :v_half_size] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                        image_syn.data[c*args.ipc:(c+1)*args.ipc, :, v_half_size:, :v_half_size] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                        image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :v_half_size, v_half_size:] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
                        image_syn.data[c*args.ipc:(c+1)*args.ipc, :, v_half_size:, v_half_size:] = downscale(imgs_real[start:end], 0.5).detach().data; start, end = end, end+args.ipc
            
        ##
        
        def get_syn_optimizer(aud_syn, img_syn):
            print(f"[Debug] aud_syn type: {type(aud_syn)}, requires_grad: {getattr(aud_syn, 'requires_grad', 'N/A')}")
            print(f"[Debug] img_syn type: {type(img_syn)}, requires_grad: {getattr(img_syn, 'requires_grad', 'N/A')}")
    
            if isinstance(aud_syn, torch.Tensor) and not isinstance(aud_syn, torch.nn.Parameter):
                aud_syn = torch.nn.Parameter(aud_syn, requires_grad=True)
            if isinstance(img_syn, torch.Tensor) and not isinstance(img_syn, torch.nn.Parameter):
                img_syn = torch.nn.Parameter(img_syn, requires_grad=True)
                
            param_groups = []
            if args.input_modality == 'a' or args.input_modality == 'av':
                param_groups += [{'params': [aud_syn], 'lr': args.lr_syn_aud}]
            
            if args.input_modality == 'v' or args.input_modality == 'av':
                param_groups += [{'params': [img_syn], 'lr': args.lr_syn_img}]
            print(f"[Debug] param_groups: {param_groups}")
            if len(param_groups) == 0:
                raise ValueError("optimizer got an empty parameter list - check input_modality and syn data.")
            return torch.optim.SGD(param_groups, momentum=0.5)
        
        # ''' training '''
        optimizer_comb = get_syn_optimizer(aud_syn, image_syn)  
        args.cf_loss_func = CFLossFunc(
            alpha_for_loss=args.alpha_for_loss, beta_for_loss=args.beta_for_loss
        )
        print('%s training begins'%get_time())
        print('%s training begins'%get_time())
        for it in range(args.Iteration+1):

            if it in eval_it_pool:
                ''' Evaluate synthetic data '''
                aud_syn_eval, image_syn_eval = None, None
                if args.input_modality == 'a' or args.input_modality == 'av':
                    aud_syn_eval = copy.deepcopy(aud_syn.detach())
                if args.input_modality == 'v' or args.input_modality == 'av':
                    image_syn_eval = copy.deepcopy(image_syn.detach())
                label_syn_eval = copy.deepcopy(label_syn.detach())
                
                aud_syn_eval1 = aud_syn_eval
                image_syn_eval1 = image_syn_eval
                if args.idm_aug:
                    if aud_syn_eval is not None:
                        aud_syn_eval1 = number_sign_augment(aud_syn_eval)
                    if image_syn_eval is not None:
                        image_syn_eval1 = number_sign_augment(image_syn_eval)

                    label_syn_eval = label_syn_eval.repeat(4)
                accs = []
                for it_eval in range(args.num_eval):
                    nets, net_eval = get_network(args)
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

            nets, _ = get_network(args)
            (net_audio, net_frame, _) = nets
            if args.input_modality == 'a' or args.input_modality == 'av':
                net_audio.to(args.device)
                net_audio.train()
                for param in list(net_audio.parameters()):
                    param.requires_grad = False
                audio_embd = net_audio.module.embed if torch.cuda.device_count() > 1 else net_audio.embed 

            if args.input_modality == 'v' or args.input_modality == 'av': 
                net_frame.to(args.device)   
                net_frame.train()        
                for param in list(net_frame.parameters()):
                    param.requires_grad = False
                image_embd = net_frame.module.embed if torch.cuda.device_count() > 1 else net_frame.embed 

            ''' Train synthetic data '''
            loss_avg = 0
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                loss_c = torch.tensor(0.0).to(args.device)
                aud_real, img_real = get_aud_images(c, args.batch_real)

                if args.input_modality == 'a' or args.input_modality == 'av':
                    aud_real = aud_real.to(args.device)
                if args.input_modality == 'v' or args.input_modality == 'av':
                    img_real = img_real.to(args.device)
                
                if args.input_modality == 'a' or args.input_modality == 'av':
                    curr_aud_syn = aud_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel[0], im_size[0][0], im_size[0][1]))
                if args.input_modality == 'v' or args.input_modality == 'av':
                    curr_img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel[1], im_size[1][0], im_size[1][1]))

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

                if args.input_modality == 'a' or args.input_modality == 'av':
                    embd_aud_real = audio_embd(aud_real).detach()
                    embd_aud_syn = audio_embd(curr_aud_syn)
                
                if args.input_modality == 'v' or args.input_modality == 'av':
                    embd_img_real = image_embd(img_real).detach()
                    embd_img_syn = image_embd(curr_img_syn)

                ## Embedding matching
                if args.input_modality == 'av':                    
                    # 1. 单模态特征匹配 (Intra-Modal Matching)
                    # 使用 NCFM 中的 match_loss，这是一种更高级的特征匹配方法，可能包含了因果公平性(CF)的考量。
                    loss_c += match_loss(embd_aud_real, embd_aud_syn, args)
                    loss_c += match_loss(embd_img_real, embd_img_syn, args)
                
                    # 2. 隐式跨模态匹配 (Implicit Cross-Modal Matching)
                    # 目标：让真实(音+视)的组合关系 与 合成(音+视)的组合关系 保持一致。
                
                    # a. 将特征向量归一化，这是计算余弦相似度的标准步骤。
                    embd_aud_real_norm = F.normalize(embd_aud_real, p=2, dim=1)
                    embd_img_real_norm = F.normalize(embd_img_real, p=2, dim=1)
                    embd_aud_syn_norm = F.normalize(embd_aud_syn, p=2, dim=1)
                    embd_img_syn_norm = F.normalize(embd_img_syn, p=2, dim=1)
                
                    # b. 通过逐元素相乘，创建模态融合后的特征表示。
                    real_combined = embd_aud_real_norm * embd_img_real_norm
                    syn_combined = embd_aud_syn_norm * embd_img_syn_norm
                
                    # c. 计算真实组合特征与合成组合特征之间的余弦相似度。
                    # 损失为 1 - cos_sim，目的是让相似度趋近于1（即角度趋近于0）。
                    cos_sim = torch.mm(real_combined, syn_combined.T)
                    loss_c += args.lam_icm * torch.mean(1 - cos_sim)
                
                
                    # 3. 跨模态差距匹配 (Cross-Modal Gap Matching)
                    # 目标：让 真实音频与合成视频的“混搭”关系 与 真实视频与合成音频的“混搭”关系 保持一致。
                    # 这能强制模型学习到不同模态间的正确“距离”或“关系”。
                
                    # a. 计算各组特征的均值。
                    aud_real_avg = torch.mean(embd_aud_real, dim=0).unsqueeze(0)
                    img_syn_avg = torch.mean(embd_img_syn, dim=0).unsqueeze(0)
                    img_real_avg = torch.mean(embd_img_real, dim=0).unsqueeze(0)
                    aud_syn_avg = torch.mean(embd_aud_syn, dim=0).unsqueeze(0)
                
                    # b. 计算两种“混搭”组合特征的余弦相似度。
                    cross_cos_lsss = torch.mm(aud_real_avg * img_syn_avg, (img_real_avg * aud_syn_avg).T)
                    loss_c += args.lam_cgm * torch.mean(1 - cross_cos_lsss)
                
                
                elif args.input_modality == 'a':
                    loss_c += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)
                
                elif args.input_modality == 'v':
                    loss_c += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)


                optimizer_comb.zero_grad()
                loss_c.backward()
                optimizer_comb.step()

                loss += loss_c.item()

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
    parser.add_argument('--alpha_for_loss', default=0.5, type=float)
    parser.add_argument('--beta_for_loss', default=0.5, type=float)
    parser.add_argument('--num_freqs', default=4090, type=int)
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

