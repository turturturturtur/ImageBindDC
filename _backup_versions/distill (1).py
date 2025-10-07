import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
import wandb
import copy
import random


from reparam_module import ReparamModule

from utils.old_data_utils import get_train_dataset, get_test_dataset, ParamDiffAug,  get_time
from utils.old_train_utils_DM import get_network, evaluate_synset_av

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# [替换] main函数中的数据加载和辅助函数定义部分

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    
    # 1. 使用您自己的函数加载AVE数据集
    channel, im_size, num_classes, _, _, dst_train = get_train_dataset(args.dataset, args)
    _, _, _, testloader = get_test_dataset(args.dataset, args)
    args.cls_num = num_classes 

    # 如果还有其他对args的修改，都放在这里...

    model_eval_pool = [args.model]
    
    # --- 然后才初始化wandb ---
    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="MTT",
               config=args,
               )
    args.dsa_param = ParamDiffAug()
    
    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    # 2. [新增] 定义您项目特有的数据获取辅助函数
    # 我们将它们直接定义在main函数内部，方便调用
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    labels_all = [dst_train[i]['label'] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    
    def get_aud_images(c, n): 
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        aud_data = dst_train[idx_shuffle]['audio'].to(args.device) if args.input_modality != 'v' else None
        img_data = dst_train[idx_shuffle]['frame'].to(args.device) if args.input_modality != 'a' else None
        return aud_data, img_data
    

# [替换] 合成数据的初始化部分

    ''' initialize the synthetic data '''
    image_syn, aud_syn = None, None
    if args.input_modality in ['a', 'av']:
        aud_syn = torch.randn(size=(num_classes*args.ipc, channel[0], im_size[0][0], im_size[0][1]), dtype=torch.float)
    if args.input_modality in ['v', 'av']:
        image_syn = torch.randn(size=(num_classes*args.ipc, channel[1], im_size[1][0], im_size[1][1]), dtype=torch.float)

    label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

    # 为可训练学习率创建一个张量
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)


    # 根据真实数据进行初始化
    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            aud_real_init, img_real_init = get_aud_images(c, args.ipc)
            if aud_syn is not None and aud_real_init is not None:
                aud_syn.data[c*args.ipc:(c+1)*args.ipc] = aud_real_init.detach().data
            if image_syn is not None and img_real_init is not None:
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = img_real_init.detach().data
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    # 将所有需要优化的合成数据张量加入参数列表
    optim_params = []
    
    if aud_syn is not None:
        aud_syn = aud_syn.detach().to(args.device).requires_grad_(True)
        optim_params.append(aud_syn)
    if image_syn is not None:
        image_syn = image_syn.detach().to(args.device).requires_grad_(True)
        optim_params.append(image_syn)

    optimizer_img = torch.optim.SGD(optim_params, lr=args.lr_img, momentum=0.5)
    optimizer_img.zero_grad()

    # 为可训练学习率创建独立的优化器
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_lr.zero_grad()
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

# [替换] 从 expert_dir 定义开始，一直到 wandb.finish() 之前
    
    # --- 步骤3.A: 加载专家轨迹 ---
    expert_dir = os.path.join(args.buffer_path, args.dataset, args.model)
    print("Expert Dir: {}".format(expert_dir))
    if not os.path.exists(expert_dir):
        raise AssertionError(f"Buffer path {expert_dir} does not exist. Please run buffer.py first.")

    # 加载所有轨迹文件到一个大的buffer中
    buffer = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        print(f"Loading buffer {n}...")
        buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    print(f"Loaded {len(buffer)} expert trajectories.")


    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}


    # ======================== 主蒸馏循环开始 ========================
    for it in range(0, args.Iteration + 1):
        save_this_it = False
        wandb.log({"Progress": it}, step=it)

        # --- 评估逻辑 (与之前类似，但现在位于循环顶部) ---
        if it in eval_it_pool:
            print('-------------------------\nEvaluation\niteration = %d' % (it))
            accs_test = []
            for it_eval in range(args.num_eval):
                # 注意：这里我们直接使用您的 evaluate_synset_av 函数
                # 它内部会创建和训练评估模型
                nets_eval, net_eval = get_network(args)
                
                # 准备评估数据
                eval_aud, eval_img = (copy.deepcopy(aud_syn.detach()) if aud_syn is not None else None, 
                                      copy.deepcopy(image_syn.detach()) if image_syn is not None else None)
                eval_lab = copy.deepcopy(label_syn.detach())
                original_dsa_strategy = args.dsa_strategy
                # 2. 临时将策略设为'none'，以安全地跳过对音频的图像增强
                args.dsa_strategy = 'none'
                
                # 3. 调用评估函数                    acc = evaluate_synset_av(nets_eval, net_eval, eval_aud, eval_img, eval_lab, testloader, args)
                    
                    # 4. 恢复原始设置，以防影响其他部分
                args.dsa_strategy = original_dsa_strategy
                original_dsa_status = args.dsa
                args.dsa = False       
                # 2. 调用评估函数
                acc = evaluate_synset_av(nets_eval, net_eval, eval_aud, eval_img, eval_lab, testloader, args)        
                # 3. 恢复原始设置，以防影响其他部分
                args.dsa = original_dsa_status

                accs_test.append(acc)
            
            acc_test_mean = np.mean(accs_test)
            acc_test_std = np.std(accs_test)
            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs_test), args.model, acc_test_mean, acc_test_std))
            # ... wandb 日志和保存逻辑可以加在这里 ...


        # --- 步骤3.B: 初始化学生网络 ---
        student_nets, student_net_eval = get_network(args)
        student_net_eval = student_net_eval.to(args.device)
        student_net_eval = ReparamModule(student_net_eval) # 使用ReparamModule包装，使其参数可微分
        student_net_eval.train()
        
        num_params = sum([np.prod(p.size()) for p in student_net_eval.parameters()])

        # --- 步骤3.C: 采样专家片段 ---
        expert_trajectory = random.choice(buffer)
        start_epoch = random.randint(0, args.max_start_epoch)
        
        # 加载专家起点和终点状态
        starting_params_dict = expert_trajectory[start_epoch]
        target_params_dict = expert_trajectory[start_epoch + args.expert_epochs]

        # 将学生网络的初始状态设置为专家的起点状态
        student_net_eval.load_state_dict(starting_params_dict)

        # 将起点和终点参数“展平”为一维向量，用于计算距离
        starting_params = torch.cat([p.data.reshape(-1) for p in student_net_eval.parameters()], 0).requires_grad_(False)
        target_params = torch.cat([p.data.reshape(-1) for p in target_params_dict.values()], 0).requires_grad_(False)
        
        # 将学生网络的参数也作为可优化的张量列表
        student_params = [p for p in student_net_eval.parameters() if p.requires_grad]

        # --- 步骤3.D: 在合成数据上“模拟训练”学生 ---
        for step in range(args.syn_steps):
            # 从合成数据中随机采样一个批次
            indices = torch.randperm(len(label_syn))[:args.batch_syn]
            
            batch_aud_syn = aud_syn[indices] if aud_syn is not None else None
            batch_img_syn = image_syn[indices] if image_syn is not None else None
            batch_lab_syn = label_syn[indices]

            # 学生网络前向传播
            output = student_net_eval(batch_aud_syn, batch_img_syn)
            
            # 处理Ensemble情况
            if args.arch_classifier == 'ensemble':
                out_a, out_v = output
                loss = criterion(out_a, batch_lab_syn) + criterion(out_v, batch_lab_syn)
            else:
                loss = criterion(output, batch_lab_syn)
            
            # 计算损失关于学生网络参数的梯度
            grad = torch.autograd.grad(loss, student_params, create_graph=True)

            # 手动执行一步SGD更新
            student_params = [p - syn_lr * g for p, g in zip(student_params, grad)]
        
        # --- 步骤3.E: 计算轨迹匹配损失 ---
        # 将模拟训练N步后的学生参数展平
        final_student_params = torch.cat([p.data.reshape(-1) for p in student_params], 0)
        
        # 计算与专家目标参数的距离
        param_loss = torch.nn.functional.mse_loss(final_student_params, target_params, reduction="sum")
        param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
        
        # 归一化损失
        grand_loss = param_loss / param_dist

        # --- 步骤3.F: 更新合成数据 ---
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        
        grand_loss.backward() # 梯度将一路穿越N步更新，传导至合成数据
        
        optimizer_img.step()
        optimizer_lr.step()
        
        # --- 日志打印 ---
        if it % 10 == 0:
            print(f"{get_time()} iter = {it:05d}, loss = {grand_loss.item():.4f}, syn_lr = {syn_lr.item():.6f}")

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(), "Synthetic_LR": syn_lr.detach().cpu()}, step=it)

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing for MTT')

    # --- 核心方法论参数 (MTT专用) ---
    parser.add_argument('--method', type=str, default='MTT', help='Specify the method to ensure correct wrapper selection')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='Path to expert trajectory buffers')
    parser.add_argument('--Iteration', type=int, default=2000, help='How many distillation steps to perform')
    parser.add_argument('--syn_steps', type=int, default=20, help='How many steps to take on synthetic data (N)')
    parser.add_argument('--expert_epochs', type=int, default=3, help='How many expert epochs the target params are (M)')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='Max epoch we can start at from expert trajectory')
    
    # --- 学习率参数 (MTT专用) ---
    parser.add_argument('--lr_img', type=float, default=1000, help='Learning rate for updating synthetic data')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='Learning rate for updating the trainable learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='Initialization for trainable learning rate')
    parser.add_argument('--lr_sound', type=float, default=1e-3, help='sound learning rate for evaluation model')
    parser.add_argument('--lr_frame', type=float, default=1e-4, help='frame learning rate for evaluation model')
    parser.add_argument('--lr_classifier', type=float, default=1e-3, help='classifier learning rate for evaluation model')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for adam optimizer in evaluation')
    
    # --- 数据与模型参数 (来自您的项目) ---
    parser.add_argument('--dataset', type=str, default='AVE', help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset')
    parser.add_argument('--model', type=str, default='AV-ConvNet', help='Name of the model architecture')
    parser.add_argument('--input_modality', type=str, default='av', help='a/v/av')
    parser.add_argument('--ipc', type=int, default=1, help='Image(s) per class')
    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"], help='Initialize synthetic data from noise or real images')

    # --- 网络结构参数 (您的get_network函数需要) ---
    parser.add_argument('--arch_sound', type=str, default='convNet', help='Sound network architecture')
    parser.add_argument('--arch_frame', type=str, default='convNet', help='Frame network architecture')
    parser.add_argument('--arch_classifier', type=str, default='ensemble', help='Classifier architecture')
    parser.add_argument('--weights_sound', type=str, default='', help='Weights for sound network')
    parser.add_argument('--weights_frame', type=str, default='', help='Weights for frame network')
    parser.add_argument('--weights_classifier', type=str, default='', help='Weights for classifier')
    
    # --- 评估参数 ---
    parser.add_argument('--eval_it', type=int, default=500, help='How often to evaluate')
    parser.add_argument('--num_eval', type=int, default=3, help='How many networks to evaluate on')
    parser.add_argument('--epoch_eval_train', type=int, default=30, help='Epochs to train a model with synthetic data')
    parser.add_argument('--batch_train', type=int, default=128, help='Batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='Batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='Batch size for syn data, None means using full-support')

    # --- [新增] 来自原始distill.py的参数，确保兼容性 ---
    parser.add_argument('--texture', action='store_true', help="distill textures instead of images")
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--subset', type=str, default='imagenette', help="ImageNet subset. This is not used for AVE dataset.")
    parser.add_argument('--res', type=int, default=128, help="resolution of ImageNet images. Not used for AVE.")
    parser.add_argument('--max_files', type=int, default=None, help="limit the number of buffer files to load")
    parser.add_argument('--max_experts', type=int, default=None, help="limit the number of experts to load")

    # --- 其他通用参数 ---
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'], help='Whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='DSA strategy')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam optimizer in evaluation')

    args = parser.parse_args()
    
    # 自动设置设备和其他参数
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa = True if args.dsa == 'True' else False
 
    
    main(args)
