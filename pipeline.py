import yaml
import model
import parsers
import torch
import torch.nn as nn
from factory import create_model


def main():
    with open("config/model/imagebind.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    model = create_model(model_cfg["name"], **model_cfg["params"])
    print(model)

if __name__ == "__main__":

    main()

