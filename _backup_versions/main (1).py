import os
import torch.nn as nn
import argparse
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from torchvision import transforms
from deepcore.utils import *
from datetime import datetime
from time import sleep
import csv
import numpy as np
import copy
from deepcore.nets.imagebind.models.imagebind_model import ModalityType
import wandb


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--selection', type=str, default="uniform", help="selection method")
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('--print_freq', '-p', default=20, type=int, help='print frequency (default: 20)')
    parser.add_argument('--fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--cross", type=str, nargs="+", default=None, help="models for cross-architecture experiments")

    # Optimizer and scheduler
    parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument("--nesterov", default=True, type=str_to_bool, help="if set nesterov")
    parser.add_argument("--scheduler", default="CosineAnnealingLR", type=str, help=
    "Learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
    parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")

    # Training
    parser.add_argument('--batch', '--batch-size', "-b", default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument("--train_batch", "-tb", default=None, type=int,
                     help="batch size for training, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument("--selection_batch", "-sb", default=None, type=int,
                     help="batch size for selection, if not specified, it will equal to batch size in argument --batch")

    # Testing
    parser.add_argument("--test_interval", '-ti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs; a value of 0 means no test will be run (default: 1)")
    parser.add_argument("--test_fraction", '-tf', type=float, default=1.,
                        help="proportion of test dataset used for evaluating the model (default: 1.)")

    # Selecting
    parser.add_argument("--selection_epochs", "-se", default=40, type=int,
                        help="number of epochs whiling performing selection on full dataset")
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                        help='momentum whiling performing selection (default: 0.9)')
    parser.add_argument('--selection_weight_decay', '-swd', default=5e-4, type=float,
                        metavar='W', help='weight decay whiling performing selection (default: 5e-4)',
                        dest='selection_weight_decay')
    parser.add_argument('--selection_optimizer', "-so", default="SGD",
                        help='optimizer to use whiling performing selection, e.g. SGD, Adam')
    parser.add_argument("--selection_nesterov", "-sn", default=True, type=str_to_bool,
                        help="if set nesterov whiling performing selection")
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.1, help='learning rate for selection')
    parser.add_argument("--selection_test_interval", '-sti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs during selection (default: 1)")
    parser.add_argument("--selection_test_fraction", '-stf', type=float, default=1.,
             help="proportion of test dataset used for evaluating the model while preforming selection (default: 1.)")
    parser.add_argument('--balance', default=True, type=str_to_bool,
                        help="whether balance selection is performed per class")

    # Algorithm
    parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use")
    parser.add_argument('--submodular_greedy', default="LazyGreedy", help="specifiy greedy algorithm for submodular optimization")
    parser.add_argument('--uncertainty', default="Entropy", help="specifiy uncertanty score to use")

    # Checkpoint and resumption
    parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')
    parser.add_argument('--resume', '-r', type=str, default='', help="path to latest checkpoint (default: do not load)")

    # Module
    parser.add_argument('--module', default="a", help="a/v/av")
    parser.add_argument('--ipc', type=int, default=None, help="image(s) per class")

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化 wandb
    wandb.init(
        project="audio-visual-deepcore",
        name=f"{args.dataset}_{args.model}_{args.selection}_frac{args.fraction}_seed{args.seed}",
        config=vars(args)
    )
    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if args.resume != "":
        # Load checkpoint
        try:
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            assert {"exp", "epoch", "state_dict", "opt_dict", "best_acc1", "rec", "subset", "sel_args"} <= set(
                checkpoint.keys())
            assert 'indices' in checkpoint["subset"].keys()
            start_exp = checkpoint['exp']
            start_epoch = checkpoint["epoch"]
        except AssertionError:
            try:
                assert {"exp", "subset", "sel_args"} <= set(checkpoint.keys())
                assert 'indices' in checkpoint["subset"].keys()
                print("=> The checkpoint only contains the subset, training will start from the begining")
                start_exp = checkpoint['exp']
                start_epoch = 0
            except AssertionError:
                print("=> Failed to load the checkpoint, an empty one will be created")
                checkpoint = {}
                start_exp = 0
                start_epoch = 0
    else:
        checkpoint = {}
        start_exp = 0
        start_epoch = 0

    best_prec1s = []
    for exp in range(start_exp, args.num_exp):
        if args.save_path != "":
            checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_epoch{epc}_{dat}_{fr}_".format(dst=args.dataset,
                                                                                         net=args.model,
                                                                                         mtd=args.selection,
                                                                                         dat=datetime.now(),
                                                                                         exp=start_exp,
                                                                                         epc=args.epochs,
                                                                                         fr=args.fraction)

        print('\n================== Exp %d ==================\n' % exp)
        print("dataset: ", args.dataset, ", model: ", args.model, ", selection: ", args.selection, ", num_ex: ",
              args.num_exp, ", epochs: ", args.epochs, ", fraction: ", args.fraction, ", seed: ", args.seed,
              ", lr: ", args.lr, ", save_path: ", args.save_path, ", resume: ", args.resume, ", device: ", args.device,
              ", checkpoint_name: " + checkpoint_name if args.save_path != "" else "", "\n", sep="")
        
        if args.module == "av":
            args_a = copy.deepcopy(args)
            args_v = copy.deepcopy(args)

            args_a.module = "a"
            args_v.module = "v"

            dataset_str = args.dataset
            dataset_av = dataset_str.split(',')
            args_a.dataset = dataset_av[0]  # 第一个数据集
            args_v.dataset = dataset_av[1]  # 第二个数据集

            if args.model != "ImageBind":
                model_av = args.model.split(',')
                args_a.model = model_av[0]
                args_v.model = model_av[1]

        if args.module != "av":
            channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
                (args.data_path)
            args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
        else:
            channel_a, im_size_a, num_classes_a, class_names_a, mean_a, std_a, dst_train_a, dst_test_a = datasets.__dict__[args_a.dataset] \
                (args.data_path)
            args_a.channel, args_a.im_size, args_a.num_classes, args_a.class_names = channel_a, im_size_a, num_classes_a, class_names_a
            channel_v, im_size_v, num_classes_v, class_names_v, mean_v, std_v, dst_train_v, dst_test_v = datasets.__dict__[args_v.dataset] \
                (args.data_path)
            args_v.channel, args_v.im_size, args_v.num_classes, args_v.class_names = channel_v, im_size_v, num_classes_v, class_names_v
            args.channel, args.im_size, args.num_classes, args.class_names = args_v.channel, args_v.im_size, args_v.num_classes, args_v.class_names
            channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = channel_v, im_size_v, num_classes_v, class_names_v, mean_v, std_v, dst_train_v, dst_test_v
        
        if args.model == "ImageBind":
            if args.module == "av":
                dst_train = {
                    ModalityType.VISION: dst_train_v, 
                    ModalityType.AUDIO: dst_train_a
                }
            elif 'a' in args.module:
                dst_train_a = {
                    ModalityType.AUDIO: dst_train_a
                }
            else:
                dst_train_a = {
                    ModalityType.VISION: dst_train_v
                }


        torch.random.manual_seed(args.seed)
            
        if "subset" in checkpoint.keys():
            subset = checkpoint['subset']
            selection_args = checkpoint["sel_args"]
        else:
            selection_args = dict(epochs=args.selection_epochs,
                                  selection_method=args.uncertainty,
                                  balance=args.balance,
                                  greedy=args.submodular_greedy,
                                  function=args.submodular
                                  )
            # print(methods.__dict__)  # 查看模块中的所有成员
            # print(args.selection)  # 确保选择的是函数名
            if args.module != "av":
                method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
                subset = method.select()
            else:
                method_a = methods.__dict__[args.selection](dst_train_a, args_a, args.fraction, args.seed, **selection_args)
                subset_a = method_a.select()
                method_v = methods.__dict__[args.selection](dst_train_v, args_v, args.fraction, args.seed, **selection_args)
                subset_v = method_v.select()

                subset = merge_av_subset(subset_a, subset_v)

        print(len(subset["indices"]))

        # Augmentation
        if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
            dst_train.transform = transforms.Compose(
                [transforms.RandomCrop(args.im_size, padding=4, padding_mode="reflect"),
                 transforms.RandomHorizontalFlip(), dst_train.transform])
        elif args.dataset == "ImageNet":
            dst_train.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        # Handle weighted subset
        if args.module != "av":
            if_weighted = "weights" in subset.keys()
            if if_weighted:
                dst_subset = WeightedSubset(dst_train, subset["indices"], subset["weights"])
            else:
                dst_subset = torch.utils.data.Subset(dst_train, subset["indices"])
        else:
            if_weighted = "weights" in subset.keys()
            if if_weighted:
                dst_subset_a = WeightedSubset(dst_train_a, subset["indices"], subset["weights"])
                dst_subset_v = WeightedSubset(dst_train_v, subset["indices"], subset["weights"])
            else:
                dst_subset_a = torch.utils.data.Subset(dst_train_a, subset["indices"])
                dst_subset_v = torch.utils.data.Subset(dst_train_v, subset["indices"])

        # BackgroundGenerator for ImageNet to speed up dataloaders
        if args.module == "av":
            if args.dataset == "ImageNet":
                train_loader_a = DataLoaderX(dst_subset_a, batch_size=args.train_batch, shuffle=True,
                                        num_workers=args.workers, pin_memory=True)
                test_loader_a = DataLoaderX(dst_test_a, batch_size=args.train_batch, shuffle=False,
                                        num_workers=args.workers, pin_memory=True)
                train_loader_v = DataLoaderX(dst_subset_v, batch_size=args.train_batch, shuffle=True,
                                        num_workers=args.workers, pin_memory=True)
                test_loader_v = DataLoaderX(dst_test_v, batch_size=args.train_batch, shuffle=False,
                                        num_workers=args.workers, pin_memory=True)
            else:
                train_loader_a = torch.utils.data.DataLoader(dst_subset_a, batch_size=args.train_batch, shuffle=True,
                                                        num_workers=args.workers, pin_memory=True)
                train_loader_v = torch.utils.data.DataLoader(dst_subset_v, batch_size=args.train_batch, shuffle=True,
                                                        num_workers=args.workers, pin_memory=True)
                test_loader_a = torch.utils.data.DataLoader(dst_test_a, batch_size=args.train_batch, shuffle=False,
                                                        num_workers=args.workers, pin_memory=True)
                test_loader_v = torch.utils.data.DataLoader(dst_test_v, batch_size=args.train_batch, shuffle=False,
                                                        num_workers=args.workers, pin_memory=True)
        else:
            if args.dataset == "ImageNet":
                train_loader = DataLoaderX(dst_subset, batch_size=args.train_batch, shuffle=True,
                                        num_workers=args.workers, pin_memory=True)
                test_loader = DataLoaderX(dst_test, batch_size=args.train_batch, shuffle=False,
                                        num_workers=args.workers, pin_memory=True)
            else:
                train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                                        num_workers=args.workers, pin_memory=True)
                test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                                        num_workers=args.workers, pin_memory=True)

        if args.model == "ImageBind":
            if args.module == "av":
                train_loader = {
                    ModalityType.VISION: train_loader_v, 
                    ModalityType.AUDIO: train_loader_a
                }
            elif 'a' in args.module:
                train_loader = {
                    ModalityType.AUDIO: train_loader
                }
            else:
                train_loader = {
                    ModalityType.VISION: train_loader
                }
        
        print(f"train loader: {train_loader}")

        # Listing cross-architecture experiment settings if specified.
        models = [args.model]
        if isinstance(args.cross, list):
            for model in args.cross:
                if model != args.model:
                    models.append(model)

        for model in models:
            if len(models) > 1:
                print("| Training on model %s" % model)
            network = nets.__dict__[model](channel, num_classes, im_size).to(args.device)

            if args.device == "cpu":
                print("Using CPU.")
            elif args.gpu is not None:
                torch.cuda.set_device(args.gpu[0])
                network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
            elif torch.cuda.device_count() > 1:
                network = nets.nets_utils.MyDataParallel(network).cuda()

            if "state_dict" in checkpoint.keys():
                # Loading model state_dict
                network.load_state_dict(checkpoint["state_dict"])

            criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

            # Optimizer
            if args.optimizer == "SGD":
                optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay, nesterov=args.nesterov)
            elif args.optimizer == "Adam":
                optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                                 weight_decay=args.weight_decay, nesterov=args.nesterov)

            # LR scheduler
            if args.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs,
                                                                       eta_min=args.min_lr)
            elif args.scheduler == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size,
                                                            gamma=args.gamma)
            else:
                scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
            scheduler.last_epoch = (start_epoch - 1) * len(train_loader)

            if "opt_dict" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["opt_dict"])

            # Log recorder
            if "rec" in checkpoint.keys():
                rec = checkpoint["rec"]
            else:
                rec = init_recorder()

            best_prec1 = checkpoint["best_acc1"] if "best_acc1" in checkpoint.keys() else 0.0

            # Save the checkpont with only the susbet.
            # if args.save_path != "" and args.resume == "":
            #     save_checkpoint({"exp": exp,
            #                      "subset": subset,
            #                      "sel_args": selection_args},
            #                     os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model
            #                                  + "_") + "unknown.ckpt"), 0, 0.)

            for epoch in range(start_epoch, args.epochs):
                # train for one epoch
                if args.model != "ImageBind":
                    train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=if_weighted)
                else:
                    train_imagebind(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=if_weighted)
                # evaluate on validation set
                if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                    if args.model != "ImageBind":
                        prec1 = test(test_loader, network, criterion, epoch, args, rec)
                    else:
                        prec1 = test_imagebind(test_loader, network, criterion, epoch, args, rec)

                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > best_prec1

                    if is_best:
                        best_prec1 = prec1
                        if args.save_path != "":
                            rec = record_ckpt(rec, epoch)
                            save_checkpoint({"exp": exp,
                                             "epoch": epoch + 1,
                                             "state_dict": network.state_dict(),
                                             "opt_dict": optimizer.state_dict(),
                                             "best_acc1": best_prec1,
                                             "rec": rec,
                                             "subset": subset,
                                             "sel_args": selection_args},
                                            os.path.join(args.save_path, checkpoint_name + (
                                                "" if model == args.model else model + "_") + "unknown.ckpt"),
                                            epoch=epoch, prec=best_prec1)

            # Prepare for the next checkpoint
            if args.save_path != "":
                try:
                    os.rename(
                        os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model + "_") +
                                     "unknown.ckpt"), os.path.join(args.save_path, checkpoint_name +
                                     ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1))
                except:
                    save_checkpoint({"exp": exp,
                                     "epoch": args.epochs,
                                     "state_dict": network.state_dict(),
                                     "opt_dict": optimizer.state_dict(),
                                     "best_acc1": best_prec1,
                                     "rec": rec,
                                     "subset": subset,
                                     "sel_args": selection_args},
                                    os.path.join(args.save_path, checkpoint_name +
                                                 ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1),
                                    epoch=args.epochs - 1,
                                    prec=best_prec1)

            print('| Best accuracy: ', best_prec1, ", on model " + model if len(models) > 1 else "", end="\n\n")
            best_prec1s.append(best_prec1)
            start_epoch = 0
            checkpoint = {}
            sleep(2)

    # Calculate average and standard deviation of best_prec1 across experiments
    average_acc = np.mean(best_prec1s)
    std_acc = np.std(best_prec1s)


    # Save the results to a CSV file
    if args.save_path != "":
        filename = os.path.join(args.save_path, f"{args.dataset}_{args.model}_{args.selection}_results.csv")
        # If the file doesn't exist, create it and write headers
        file_exists = os.path.exists(filename)
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(["Dataset", "Model", "Selection Method", "Fraction", "Num Experiments", "Timestamp", 
                                "Experiment Index", "Best Accuracy", "Average Accuracy", "Std Accuracy"])
            
            # Write each experiment's result
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for exp_idx, best_acc in enumerate(best_prec1s):
                writer.writerow([args.dataset, args.model, args.selection, args.fraction, args.num_exp, timestamp,
                                exp_idx, best_acc, average_acc, std_acc])

        print(f"Precisions: {best_prec1s}")
        print(f"Average best accuracy: {average_acc:.4f}, Standard Deviation: {std_acc:.4f}")


if __name__ == '__main__':
    main()
