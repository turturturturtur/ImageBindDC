import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import ModelBuilder
from utils.train_utils import NetWrapper, adjust_learning_rate, NetWrapperimagebind
from utils.data_utils import CombTensorDataset, DiffAugment

TARGET_WIDTH = 204  # Define the target width for audio data

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
        input_size=1024)  # Imagebind default embedding size is 1024

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
    return torch.optim.Adam(param_groups, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)  

def preprocess_audio(audio):
    """
    Unifies audio tensor dimensions to [B, C, 1, H, W] and pads/crops to TARGET_WIDTH.
    """
    if audio.dim() == 3:
        audio = audio.unsqueeze(1).unsqueeze(1)
    elif audio.dim() == 4:
        audio = audio.unsqueeze(2)
    
    w = audio.shape[-1]
    if w < TARGET_WIDTH:
        pad_w = TARGET_WIDTH - w
        audio = F.pad(audio, (0, pad_w), "constant", 0)
    elif w > TARGET_WIDTH:
        audio = audio[..., :TARGET_WIDTH]
    return audio

def evaluate(netWrapper, loader, args):
    criterion = nn.CrossEntropyLoss()
    netWrapper.eval()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for batch_data in loader:
            audio, frame = None, None
            if args.input_modality in ['a', 'av']:
                audio = preprocess_audio(batch_data['audio'].float().to(args.device))
            if args.input_modality in ['v', 'av']:
                frame = batch_data['frame'].float().to(args.device)
            gt = batch_data['label'].to(args.device)

            if args.arch_classifier == 'ensemble':
                out_a, out_v = netWrapper.forward(audio, frame)
                err = criterion(out_a, gt) + criterion(out_v, gt)
                preds = (F.softmax(out_a, dim=1) + F.softmax(out_v, dim=1)) / 2
            else:
                preds = netWrapper.forward(audio, frame)
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
    netWrapper.train()
    correct, total, total_loss = 0, 0, 0.0

    for batch_data in loader:
        audio, frame = None, None
        if args.input_modality in ['a', 'av']:
            audio = preprocess_audio(batch_data['audio'].float().to(args.device))
            # Optional differentiable augmentation
            audio_squeeze = audio.squeeze(2)
            audio_squeeze = DiffAugment(audio_squeeze, args.dsa_strategy, param=args.dsa_param)
            audio = audio_squeeze.unsqueeze(2)
        if args.input_modality in ['v', 'av']:
            frame = batch_data['frame'].float().to(args.device)
            frame = DiffAugment(frame, args.dsa_strategy, param=args.dsa_param)
        gt = batch_data['label'].to(args.device)

        optimizer.zero_grad()
        if args.arch_classifier == 'ensemble':
            out_a, out_v = netWrapper.forward(audio, frame)
            err = criterion(out_a, gt) + criterion(out_v, gt)
            preds = (F.softmax(out_a, dim=1) + F.softmax(out_v, dim=1)) / 2
        else:
            preds = netWrapper.forward(audio, frame)
            err = criterion(preds, gt)

        err.backward()
        optimizer.step()

        _, predicted = torch.max(preds.data, 1)
        total += preds.size(0)
        correct += (predicted == gt).sum().item()
        total_loss += err.item()

    average_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return average_loss, accuracy

def evaluate_synset_av(nets, net_eval, auds_train, images_train, labels_train, testloader, args):
    reset_params(args)
    net_eval = net_eval.to(args.device)

    optimizer = create_optimizer_imagebind(nets, args) if len(nets) == 2 else create_optimizer(nets, args)
    
    # Preprocess and move data to device
    if args.input_modality in ['a', 'av']:
        auds_train = preprocess_audio(auds_train).to(args.device)
    if args.input_modality in ['v', 'av']:
        images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)

    dst_train = CombTensorDataset(auds_train, images_train, labels_train, args)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_syn, shuffle=True, num_workers=0)

    for e in range(args.epoch_eval_train):
        train(net_eval, trainloader, optimizer, args)
        if e in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    _, val_acc = evaluate(net_eval, testloader, args)
    return round(val_acc, 2)

def reset_params(args):
    args.weights_sound = ''
    args.weights_frame = ''
    args.weights_classifier = ''

    args.lr_sound = 1e-3
    args.lr_frame = 1e-4
    args.lr_classifier = 1e-3