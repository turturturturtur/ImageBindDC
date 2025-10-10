import torch
from abc import ABC, abstractmethod

class BaseLoss(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *input):
        pass