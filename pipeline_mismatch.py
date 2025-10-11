import yaml
import model
import dataset
import loss
import argparse
import torch
import torch.nn as nn
from factory import create_model, create_dataset, create_loss
from utils import read_cfg, set_seeds, get_img_transform, get_syn_data, Trainer, get_syn_optimizer, ClassSampler
from torchvision.transforms import v2
from tqdm import tqdm


def main(args):
    # 读取实验配置
    exp_cfg = read_cfg(args.exp_config)
    model_cfg = read_cfg(args.model_config)
    set_seeds(exp_cfg.get("seed", args.seed))

    # 数据增强配置
    img_transform = None
    if exp_cfg.get("img_aug",False):
        img_transform = get_img_transform()

    # 创建数据集
    dataset = create_dataset(exp_cfg.get("dataset"))
    dst_train = dataset.build(mode='train', transform=img_transform)
    dst_syn = dataset.build(mode='train', transform=img_transform)
    dst_test = dataset.build(mode='test')
    dst_syn = get_syn_data(dst_train=dst_train, dst_syn_container=dst_syn,ipc=exp_cfg.get("ipc"), mode='normal')

    # 创建模型
    model_cfg["params"]['extra_params']["num_classes"] = dst_train.num_classes
    model_teacher = create_model(model_cfg.get("name"),**model_cfg.get("params")).to('cuda')
    model_student = create_model(model_cfg.get("name"),**model_cfg.get("params")).to('cuda')


    # 创建损失函数
    criterion = create_loss(exp_cfg.get("loss"))

    # 创建优化器
    optimizer = get_syn_optimizer(dst_syn, dataset.input_modality, exp_cfg)

    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=exp_cfg.get("batch_size", 16), shuffle=True, pin_memory=True)
    syn_loader = torch.utils.data.DataLoader(dst_syn, batch_size=exp_cfg.get("batch_size", 16), shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dst_test, batch_size=exp_cfg.get("batch_size", 16), shuffle=False, pin_memory=True)

    # 创建真实数据采样器
    real_data_sampler = ClassSampler(dataset=dst_train)

    # 创建训练器
    trainer = Trainer(
        model=model_teacher,       
        optimizer=optimizer,
        loss_fn=criterion,
        train_loader=syn_loader,     
        val_loader=test_loader,  
        synthetic_dataset=dst_syn,   
        real_dataset=dst_train,
        real_batch_size=exp_cfg.get("batch_real", 128),
        real_sampler=real_data_sampler,
        epochs=exp_cfg.get("epochs"),
        val_train_epochs=exp_cfg.get("epoch_eval_train", 5),
        augment_transform=img_transform,
        mismatch_rate=args.mismatch_rate,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, default="config/experiment/distillation.yaml")
    parser.add_argument("--model_config", type=str, default="config/model/imagebind.yaml")
    parser.add_argument("--mismatch_rate", type=float)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

