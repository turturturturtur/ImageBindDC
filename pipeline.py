import yaml
import model
import argparse
import torch
import torch.nn as nn
from factory import create_model, create_dataset
from utils import read_cfg, set_seeds
from dataset import AVEBuilder, dataset_mapping


def main(args):
    
    # 读取实验配置
    exp_cfg = read_cfg(args.exp_config)
    set_seeds(exp_cfg.get("seed", 42))

    dataset = create_dataset(exp_cfg.get("dataset"))
    dst_train = dataset.build(mode='train')
    dst_test = dataset.build(mode='test')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, default="config/experiment/distillation.yaml")
    args = parser.parse_args()
    main(args)

