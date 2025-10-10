import yaml
import model
import argparse
import torch
import torch.nn as nn
from factory import create_model, create_dataset, create_loss
from utils import read_cfg, set_seeds, get_img_transform, get_syn_data
from dataset import AVEBuilder, dataset_mapping
from torchvision.transforms import v2
from tqdm import tqdm

def train_syn_data(model_teacher, dst_train, dst_syn, exp_cfg):
    pass



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
    dst_test = dataset.build(mode='test')
    dst_syn = get_syn_data(dst_train=dst_train, ipc=exp_cfg.get("ipc"))

    # 创建模型
    model_cfg["params"]['extra_params']["num_classes"] = dst_train.num_classes
    model_teacher = create_model(model_cfg.get("name"),**model_cfg.get("params")).to('cuda')
    if torch.cuda.device_count() > 1:
        model_teacher = nn.DataParallel(model_teacher)

    # 创建损失函数
    criterion = create_loss(exp_cfg.get("loss"),**exp_cfg.get("params"))

    # single test
    #
    # batch_size = 4 
    # inputs = {
    #     "audio": dst_train[:batch_size]["audio"],
    #     "image": dst_train[:batch_size]["frame"],
    # }
    # model_teacher.forward(inputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, default="config/experiment/distillation.yaml")
    parser.add_argument("--model_config", type=str, default="config/model/imagebind.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

