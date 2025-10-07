import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import ModelBuilder
from utils.train_utils import NetWrapper, adjust_learning_rate, NetWrapperimagebind
from utils.data_utils import CombTensorDataset, DiffAugment

def get_network(args):
    builder = ModelBuilder()

    net_sound = builder.build_sound(
        arch=args.arch_sound,
        weights=args.weights_sound)
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        weights=args.weights_frame)
    net_classifier = builder.build_classifier(
        arch=args.arch_classifier,
        cls_num=args.cls_num,
        weights=args.weights_classifier,
        input_modality=args.input_modality)

    if args.input_modality == 'av':
        nets = (net_sound, net_frame, net_classifier)
    elif args.input_modality == 'a':
        nets = (net_sound, None, net_classifier)
    elif args.input_modality == 'v':
        nets = (None, net_frame, net_classifier)    
    netWrapper = NetWrapper(args, nets)
    return nets, netWrapper

def get_network_imagebind(args, pretrained=True):
    builder = ModelBuilder()

    net_imagebind = builder.build_imagebind(arch=args.arch_frame, pretrained=pretrained)
    net_classifier = builder.build_classifier(
        arch=args.arch_classifier,
        cls_num=args.cls_num,
        weights=args.weights_classifier,
        input_modality=args.input_modality,
        input_size=1024)

    if args.input_modality == 'av':
        nets = (net_imagebind, net_classifier)
    
    netWrapper = NetWrapperimagebind(args, nets)
    return nets, netWrapper

def create_optimizer(nets, args):
    (net_sound, net_frame, net_classifier) = nets
    param_groups = [{'params': net_classifier.parameters(), 'lr': args.lr_classifier}]
    if net_sound is not None:
        param_groups += [{'params': net_sound.parameters(), 'lr': args.lr_sound}]
    if net_frame is not None:
        param_groups += [{'params': net_frame.parameters(), 'lr': args.lr_frame}]
    return torch.optim.Adam(param_groups, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)  

def create_optimizer_imagebind(nets, args):
    (net_imagebind, net_classifier) = nets
    param_groups = [{'params': net_classifier.parameters(), 'lr': args.lr_classifier}]
    # if net_sound is not None:
    #     param_groups += [{'params': net_sound.parameters(), 'lr': args.lr_sound}]
    # if net_frame is not None:
    #     param_groups += [{'params': net_frame.parameters(), 'lr': args.lr_frame}]
    # if net_imagebind is not None:
    #     param_groups += [{'params': net_imagebind.parameters(), 'lr': args.lr_sound}]
    return torch.optim.Adam(param_groups, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)  

def evaluate(netWrapper, loader, args):
    criterion = nn.CrossEntropyLoss()

    # switch to eval mode
    netWrapper.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for i, batch_data in enumerate(loader):

        audio, frame = None, None
        if args.input_modality == 'a' or args.input_modality == 'av':
            audio = batch_data['audio'].float().to(args.device)
        
        if args.input_modality == 'v' or args.input_modality == 'av':
            frame = batch_data['frame'].float().to(args.device)
        
        gt = batch_data['label'].to(args.device)
        
        # forward pass
        if args.arch_classifier == 'ensemble':
            out_a, out_v = netWrapper.forward(audio, frame)
            err = criterion(out_a, gt) + criterion(out_v, gt)
            preds_a = F.softmax(out_a, dim=1)
            preds_v = F.softmax(out_v, dim=1)
            preds = (preds_a + preds_v) / 2
        else:
            preds = netWrapper(audio, frame)
            err = criterion(preds, gt)

        _, predicted = torch.max(preds.data, 1)
        total += preds.size(0)
        correct += (predicted == gt).sum().item()

        total_loss += err.item()

    acc = 100 * correct / total
    average_loss = total_loss / len(loader)
    return average_loss, acc

def train(netWrapper, loader, optimizer, args):
    torch.set_grad_enabled(True) 
    criterion = nn.CrossEntropyLoss()

    # switch to train mode
    netWrapper.train()
    correct = 0
    total = 0
    total_loss = 0.0

    # main loop
    torch.cuda.synchronize()
    for i, batch_data in enumerate(loader):
        torch.cuda.synchronize()

        audio, frame = None, None
        if args.input_modality == 'a' or args.input_modality == 'av':
            audio = batch_data['audio'].float().to(args.device)
            audio = audio.squeeze(2)
            audio = DiffAugment(audio, args.dsa_strategy, param=args.dsa_param)
            audio = audio.unsqueeze(2)

        if args.input_modality == 'v' or args.input_modality == 'av':
            frame = batch_data['frame'].float().to(args.device)
            frame = DiffAugment(frame, args.dsa_strategy, param=args.dsa_param)
        gt = batch_data['label'].to(args.device)

        # forward pass
        netWrapper.zero_grad()
        if args.arch_classifier == 'ensemble':
            out_a, out_v = netWrapper.forward(audio, frame)
            err = criterion(out_a, gt) + criterion(out_v, gt)
            preds_a = F.softmax(out_a, dim=1)
            preds_v = F.softmax(out_v, dim=1)
            preds = (preds_a + preds_v) / 2
        else:
            preds = netWrapper.forward(audio, frame)
            err = criterion(preds, gt)

        _, predicted = torch.max(preds.data, 1)
        total += preds.size(0)
        correct += (predicted == gt).sum().item()

        # backward
        err.backward()
        optimizer.step()

        torch.cuda.synchronize()
        total_loss += err.item()
    
    average_loss = total_loss / len(loader)
    accuracy = correct*100 / total
    return average_loss, accuracy

def distillation_loss(student_output, teacher_output, labels, alpha=0.7, T=3):

    ce_loss = F.cross_entropy(student_output, labels)
    
    soft_log_probs = F.log_softmax(student_output/T, dim=1)
    soft_targets = F.softmax(teacher_output/T, dim=1).detach()  # 阻断梯度传播
    
    kld_loss = F.kl_div(soft_log_probs, soft_targets, reduction='batchmean') * (T**2)

    return alpha * kld_loss + (1 - alpha) * ce_loss

def distillation_loss_v2(student_output, teacher_output, labels, alpha=0.7, T=5):

    
    soft_log_probs = F.log_softmax(student_output/T, dim=1)
    soft_targets = F.softmax(teacher_output/T, dim=1).detach()  # 阻断梯度传播
    
    kld_loss = F.kl_div(soft_log_probs, soft_targets, reduction='batchmean') * (T**2)

    return kld_loss

def train_softlabels(netWrapper, loader, optimizer, args):
    torch.set_grad_enabled(True) 
    criterion = nn.CrossEntropyLoss()

    # switch to train mode
    netWrapper.train()
    correct = 0
    total = 0
    total_loss = 0.0

    # main loop
    torch.cuda.synchronize()
    for i, batch_data in enumerate(loader):
        torch.cuda.synchronize()

        audio, frame = None, None
        if args.input_modality == 'a' or args.input_modality == 'av':
            audio = batch_data['audio'].float().to(args.device)
            audio = DiffAugment(audio, args.dsa_strategy, param=args.dsa_param)

        if args.input_modality == 'v' or args.input_modality == 'av':
            frame = batch_data['frame'].float().to(args.device)
            frame = DiffAugment(frame, args.dsa_strategy, param=args.dsa_param)
        gt = batch_data['label'].to(args.device)

        preds_a_teacher = batch_data['preds_a_teacher'].to(args.device)
        preds_v_teacher = batch_data['preds_v_teacher'].to(args.device)
        
        # forward pass
        netWrapper.zero_grad()
        if args.arch_classifier == 'ensemble':
            out_a, out_v = netWrapper.forward(audio, frame)
            err = criterion(out_a, gt) + criterion(out_v, gt)
            # err = distillation_loss(out_a, preds_a_teacher, gt, args.alpha) + distillation_loss(out_v, preds_v_teacher, gt, args.alpha)
            preds_a = F.softmax(out_a, dim=1)
            preds_v = F.softmax(out_v, dim=1)
            preds = (preds_a + preds_v) / 2
            err += 2 * distillation_loss_v2(preds, preds_a_teacher, gt, args.alpha)
        else:
            preds = netWrapper.forward(audio, frame)
            err = criterion(preds, gt)
        
        

        _, predicted = torch.max(preds.data, 1)
        total += preds.size(0)
        correct += (predicted == gt).sum().item()

        # backward
        err.backward()
        optimizer.step()

        torch.cuda.synchronize()
        total_loss += err.item()
    
    average_loss = total_loss / len(loader)
    accuracy = correct*100 / total
    return average_loss, accuracy

def evaluate_synset_av(nets, net_eval, auds_train, images_train, labels_train, testloader, args):
    reset_params(args)
    print('#######################################')
    print("images_train shape:", images_train.shape)

    net_eval = net_eval.to(args.device)
    if len(nets) == 2:
        optimizer = create_optimizer_imagebind(nets, args)
    else: 
        optimizer = create_optimizer(nets, args)
    
    if args.input_modality == 'av' or args.input_modality == 'v':    
        images_train = images_train.to(args.device)
    if args.input_modality == 'av' or args.input_modality == 'a':
        auds_train = auds_train.to(args.device)
    labels_train = labels_train.to(args.device)

    dst_train = CombTensorDataset(auds_train, images_train, labels_train, args)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_syn, shuffle=True, num_workers=0)
    for e in range(args.epoch_eval_train):
        train_loss, train_acc = train(net_eval, trainloader, optimizer, args)
        # print(f'Epoch {e+1}/{args.epoch_eval_train} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}')

        if e in args.lr_steps:
            adjust_learning_rate(optimizer, args)
    
    val_loss, val_acc = evaluate(net_eval, testloader, args)
    val_acc = round(val_acc, 2)
    return val_acc

def evaluate_synset_av(nets, net_eval, auds_train, images_train, labels_train, testloader, args, ):
    reset_params(args)

    net_eval = net_eval.to(args.device)
    if len(nets) == 2:
        optimizer = create_optimizer_imagebind(nets, args)
    else: 
        optimizer = create_optimizer(nets, args)
    
    if args.input_modality == 'av' or args.input_modality == 'v':    
        images_train = images_train.to(args.device)
    if args.input_modality == 'av' or args.input_modality == 'a':
        auds_train = auds_train.to(args.device)
    labels_train = labels_train.to(args.device)

    dst_train = CombTensorDataset(auds_train, images_train, labels_train, args)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_syn, shuffle=True, num_workers=0)
    for e in range(args.epoch_eval_train):
        train_loss, train_acc = train(net_eval, trainloader, optimizer, args)
        # print(f'Epoch {e+1}/{args.epoch_eval_train} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}')

        if e in args.lr_steps:
            adjust_learning_rate(optimizer, args)
    
    val_loss, val_acc = evaluate(net_eval, testloader, args)
    val_acc = round(val_acc, 2)
    return val_acc

# def evaluate_synset_av_softlabels(nets, net_eval, auds_train, images_train, labels_train, testloader, args, soft_labels):
#     reset_params(args)

#     net_eval = net_eval.to(args.device)
#     if len(nets) == 2:
#         optimizer = create_optimizer_imagebind(nets, args)
#     else: 
#         optimizer = create_optimizer(nets, args)
    
#     if args.input_modality == 'av' or args.input_modality == 'v':    
#         images_train = images_train.to(args.device)
#     if args.input_modality == 'av' or args.input_modality == 'a':
#         auds_train = auds_train.to(args.device)
#     labels_train = labels_train.to(args.device)

#     dst_train = CombTensorDataset_softlabels(auds_train, images_train, labels_train, args, soft_labels)
#     trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_syn, shuffle=True, num_workers=0)
#     for e in range(args.epoch_eval_train):
#         train_loss, train_acc = train_softlabels(net_eval, trainloader, optimizer, args)
#         # print(f'Epoch {e+1}/{args.epoch_eval_train} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}')

#         if e in args.lr_steps:
#             adjust_learning_rate(optimizer, args)
    
#     val_loss, val_acc = evaluate(net_eval, testloader, args)
#     val_acc = round(val_acc, 2)
#     return val_acc

def get_softlabels(nets, net_eval, auds_train, images_train, labels_train, testloader, args):
    net_eval = net_eval.to(args.device)
    
    images_train = images_train.to(args.device)
    auds_train = auds_train.to(args.device)
    labels_train = labels_train.to(args.device)
    criterion = nn.CrossEntropyLoss()
    dst_train = CombTensorDataset(auds_train, images_train, labels_train, args)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_syn, shuffle=False, num_workers=0)
    netWrapper = net_eval
    loader = trainloader
    all_preds_a = []
    all_preds_v = []
    with torch.no_grad():
        for i, batch_data in enumerate(loader):

            audio, frame = None, None
            if args.input_modality == 'a' or args.input_modality == 'av':
                audio = batch_data['audio'].float().to(args.device)
                audio = DiffAugment(audio, args.dsa_strategy, param=args.dsa_param)

            if args.input_modality == 'v' or args.input_modality == 'av':
                frame = batch_data['frame'].float().to(args.device)
                frame = DiffAugment(frame, args.dsa_strategy, param=args.dsa_param)

            out_a, out_v = netWrapper.forward(audio, frame)
            preds_a = F.softmax(out_a, dim=1)
            preds_v = F.softmax(out_v, dim=1)
            preds = (preds_a + preds_v) / 2
            # all_preds_a.append(preds_a.detach().cpu())  # 移出GPU并断开计算图
            # all_preds_v.append(preds_v.detach().cpu())
            all_preds_a.append(preds.detach().cpu())  # 移出GPU并断开计算图
            all_preds_v.append(preds.detach().cpu())
            
            
    final_preds_a = torch.cat(all_preds_a, dim=0)
    final_preds_v = torch.cat(all_preds_v, dim=0)
    save_dict = {
        'preds_a': final_preds_a,
        'preds_v': final_preds_v
    }
    return save_dict

    
    
    
def reset_params(args):
    args.weights_sound = ''
    args.weights_frame = ''
    args.weights_classifier = ''

    args.lr_sound = 1e-3
    args.lr_frame = 1e-4
    args.lr_classifier = 1e-3
