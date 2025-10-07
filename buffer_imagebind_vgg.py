import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.data_utils import get_train_dataset_origin as get_train_dataset
from utils.data_utils import get_test_dataset_origin as get_test_dataset
from utils.data_utils import ParamDiffAug

from utils.train_utils_DM import get_network_imagebind
from utils.old_train_utils_DM import  train_expert_epoch, test_expert_epoch
from torch.utils.data import TensorDataset

import copy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):


    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    # 1. 加载AVE数据集
    channel, im_size, num_classes, _, _, dst_train = get_train_dataset(args.dataset, args)
    _, _, _, testloader = get_test_dataset(args.dataset, args)
    args.cls_num = num_classes # 将 num_classes 改为 cls_num

    # 2. 从dst_train创建训练数据加载器
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=4)

    print('Hyper-parameters: \n', args.__dict__)

    # 3. 创建保存轨迹的目录 (逻辑保留)
    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 4. 定义损失函数
    criterion = nn.CrossEntropyLoss().to(args.device)
    trajectories = []

    for it in range(0, args.num_experts):
        print(f"\n================== Training Expert {it} ==================\n")
        
        _, teacher_net = get_network_imagebind(args)
        teacher_net.to(args.device)
        
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=args.lr_teacher, momentum=0.9, weight_decay=0.0005)
        
        timestamps = []
        timestamps.append([p.detach().clone() for p in teacher_net.parameters()])

        for e in range(args.train_epochs):
            # 直接调用从utils导入的函数
            train_loss, train_acc = train_expert_epoch(teacher_net, trainloader, teacher_optim, criterion, args)
            test_loss, test_acc = test_expert_epoch(teacher_net, testloader, criterion, args)

            print(f"Expert: {it}\tEpoch: {e+1}/{args.train_epochs}\tTrain Acc: {train_acc:.2f}%\tTest Acc: {test_acc:.2f}%")

            timestamps.append([p.detach().clone() for p in teacher_net.parameters()])

        trajectories.append(timestamps)
        # 6. 周期性地保存轨迹到文件 (逻辑保留)
        if len(trajectories) == args.save_interval or it == args.num_experts - 1:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # --- 核心参数 (来自 buffer.py) ---
    parser.add_argument('--num_experts', type=int, default=10, help='Number of experts to train')
    parser.add_argument('--train_epochs', type=int, default=50, help='Number of epochs to train each expert')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='Learning rate for training expert networks')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='Path to save expert trajectory buffers')
    parser.add_argument('--save_interval', type=int, default=10, help='Save trajectories every X experts')

    # --- 数据与模型参数 (来自您的项目) ---
    parser.add_argument('--dataset', type=str, default='VGG_subset', help='dataset')
    parser.add_argument('--model', type=str, default='AV-ConvNet', help='name of the model architecture for path saving')
    parser.add_argument('--input_modality', type=str, default='av', help='a/v/av')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    
    # 网络结构参数 (您的 get_network 函数需要它们)
    
    parser.add_argument('--weights_sound', type=str, default='', help='weights for sound network')
    parser.add_argument('--weights_frame', type=str, default='', help='weights for frame network')
    parser.add_argument('--weights_classifier', type=str, default='', help='weights for classifier')
    # 其他通用参数
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--arch_frame', type=str, default='imagebind_huge', help='ImageBind model architecture')
    parser.add_argument('--arch_classifier', type=str, default='imagebind_head', help='classifier architecture')
    args = parser.parse_args()
    
    # 自动设置设备
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    main(args)

