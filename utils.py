import time, torch
from argparse import ArgumentTypeError
from prefetch_generator import BackgroundGenerator
import csv
import os
from datetime import datetime
import numpy as np



class WeightedSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]], self.weights[[i for i in idx]]
        return self.dataset[self.indices[idx]], self.weights[idx]


def train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted: bool = False):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to train mode
    network.train()

    end = time.time()
    for i, contents in enumerate(train_loader):
        optimizer.zero_grad()
        if if_weighted:
            target = contents[0][1].to(args.device)
            input = contents[0][0].to(args.device)

            # Compute output
            output = network(input)
            weights = contents[1].to(args.device).requires_grad_(False)
            loss = torch.sum(criterion(output, target) * weights) / torch.sum(weights)
        else:
            target = contents[1].to(args.device)
            input = contents[0].to(args.device)

            # Compute output
            output = network(input)
            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

    record_train_stats(rec, epoch, losses.avg, top1.avg, optimizer.state_dict()['param_groups'][0]['lr'])
    wandb.log({
        "train/loss": losses.avg,
        "train/acc": top1.avg,
        "train/lr": optimizer.state_dict()['param_groups'][0]['lr'],
        "epoch": epoch
    })

def test(test_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.to(args.device)
        input = input.to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(input)

            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    network.no_grad = False

    record_test_stats(rec, epoch, losses.avg, top1.avg)
    wandb.log({
        "test/loss": losses.avg,
        "test/acc": top1.avg,
        "epoch": epoch
    })
    return top1.avg


def train_av(train_loader_a, train_loader_v, network_a, network_v, criterion, optimizer_a, optimizer_v, scheduler_a, scheduler_v, epoch, args, rec, if_weighted: bool = False):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to train mode
    network_a.train()
    network_v.train()

    end = time.time()
    for (i, (content_a, content_v)) in enumerate(zip(train_loader_a, train_loader_v)):
        optimizer_a.zero_grad()
        optimizer_v.zero_grad()
        if if_weighted:
            target = content_a[0][1].to(args.device)

            input_a = content_a[0][0].to(args.device)
            input_v = content_v[0][0].to(args.device)

            # Compute output
            output_a = network_a(input_a)
            output_v = network_v(input_v)
            weights_a = content_a[1].to(args.device).requires_grad_(False)
            weights_v = content_v[1].to(args.device).requires_grad_(False)
            output = (output_a + output_v) / 2
            weights = (weights_a + weights_v) / 2
            loss = torch.sum(criterion(output, target) * weights) / torch.sum(weights)
        else:
            target = content_a[1].to(args.device)

            input_a = content_a[0].to(args.device)
            input_v = content_v[0].to(args.device)

            # Compute output
            output_a = network_a(input_a)
            output_v = network_v(input_v)

            output = (output_a + output_v) / 2

            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]

        # print(f"prec1: {prec1}")
        
        losses.update(loss.data.item(), input_a.size(0))
        top1.update(prec1.item(), input_a.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer_a.step()
        optimizer_v.step()
        scheduler_a.step()
        scheduler_v.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader_a), batch_time=batch_time,
                loss=losses, top1=top1))

    record_train_stats(rec, epoch, losses.avg, top1.avg, optimizer_a.state_dict()['param_groups'][0]['lr'])
    wandb.log({
        "train/loss": losses.avg,
        "train/acc": top1.avg,
        "train/lr": optimizer_a.state_dict()['param_groups'][0]['lr'],
        "epoch": epoch
    })

def test_av(test_loader_a, test_loader_v, network_a, network_v, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    network_a.eval()
    network_v.eval()
    network_a.no_grad = True
    network_v.no_grad = True

    end = time.time()
    for (i, ((input_a, target_a), (input_v, target_v))) in enumerate(zip(test_loader_a, test_loader_v)):
        target = target_a.to(args.device)
        input_a = input_a.to(args.device)
        input_v = input_v.to(args.device)

        # Compute output
        with torch.no_grad():
            output_a = network_a(input_a)
            output_v = network_v(input_v)

            output = (output_a + output_v) / 2

            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input_a.size(0))
        top1.update(prec1.item(), input_a.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader_a), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    network_a.no_grad = False
    network_v.no_grad = False

    record_test_stats(rec, epoch, losses.avg, top1.avg)
    wandb.log({
        "test/loss": losses.avg,
        "test/acc": top1.avg,
        "epoch": epoch
    })
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str_to_bool(v):
    # Handle boolean type in arguments.
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def save_checkpoint(state, path, epoch, prec):
    print("=> Saving checkpoint for epoch %d, with Prec@1 %f." % (epoch, prec))
    torch.save(state, path)


def init_recorder():
    from types import SimpleNamespace
    rec = SimpleNamespace()
    rec.train_step = []
    rec.train_loss = []
    rec.train_acc = []
    rec.lr = []
    rec.test_step = []
    rec.test_loss = []
    rec.test_acc = []
    rec.ckpts = []
    return rec


def record_train_stats(rec, step, loss, acc, lr):
    rec.train_step.append(step)
    rec.train_loss.append(loss)
    rec.train_acc.append(acc)
    rec.lr.append(lr)
    return rec


def record_test_stats(rec, step, loss, acc):
    rec.test_step.append(step)
    rec.test_loss.append(loss)
    rec.test_acc.append(acc)
    return rec


def record_ckpt(rec, step):
    rec.ckpts.append(step)
    return rec


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

def save_experiment_results(filename, accuracies, average_acc, std_acc, args):
    # 如果文件不存在，则创建文件并写入表头
    file_exists = os.path.exists(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # 如果文件不存在，则写入表头
        if not file_exists:
            writer.writerow(["Dataset", "Model", "Fraction", "Module", "Num_Experiments", "Timestamp", 
                             "Exp_Index", "Accuracy", "Average_Accuracy", "Std_Accuracy"])
        
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 记录每次实验的结果
        for exp_idx, acc in enumerate(accuracies):
            writer.writerow([args.dataset, args.model, args.fraction, args.selection, args.num_exp, timestamp,
                             exp_idx, acc, average_acc, std_acc])

def merge_av_subset(subset_a, subset_v):
    n = len(subset_v["indices"])
    top_examples = []
    scores = []

    for i in range(n):
        # 随机选择来源于 subset_v 或 subset_a 的一个索引
        selected_from_v = np.random.choice([True, False])  # True -> 从 subset_v 选择，False -> 从 subset_a 选择
        if selected_from_v:
            top_examples.append(subset_v["indices"][i])
            scores.append(subset_v["scores"][i])
        else:
            top_examples.append(subset_a["indices"][i])
            scores.append(subset_a["scores"][i])
            
    return {"indices": top_examples, "scores": scores}
    

def train_imagebind(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted: bool = False):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to train mode
    network.train()

    end = time.time()

    train_loader_v = train_loader['vision']
    train_loader_a = train_loader['audio']

    i = 0
    for (contents_v, contents_a) in zip(train_loader_v, train_loader_a):
        i = i + 1
        optimizer.zero_grad()
        if if_weighted:
            target = contents[0][1].to(args.device)
            input = contents[0][0].to(args.device)

            # Compute output
            output = network(input)
            weights = contents[1].to(args.device).requires_grad_(False)
            loss = torch.sum(criterion(output, target) * weights) / torch.sum(weights)
        else:
            target = contents_v[1].to(args.device)

            vision_input = contents_v[0].to(args.device)  # 视觉输入
            audio_input = contents_a[0].to(args.device)   # 音频输入

            input = {
                'vision': vision_input,  # 视觉模态的输入
                'audio': audio_input     # 音频模态的输入
            }

            # Compute output
            output = network(input)
            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), vision_input.size(0))
        top1.update(prec1.item(), vision_input.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

    record_train_stats(rec, epoch, losses.avg, top1.avg, optimizer.state_dict()['param_groups'][0]['lr'])


def test_imagebind(test_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.to(args.device)
        input = input.to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(input)

            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    network.no_grad = False

    record_test_stats(rec, epoch, losses.avg, top1.avg)
    return top1.avg
