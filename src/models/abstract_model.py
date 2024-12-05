import torch
from torch import nn
from abc import ABCMeta, abstractmethod

#Inherit from both
class AbstractModelMeta(nn.Module, metaclass=ABCMeta):
    pass


#define a model template
class AbstractModel(AbstractModelMeta):
    model: nn.Module

    def __init__(self, d_features: int, device: torch.device):
        super().__init__()
        self.d_features = d_features
        self.device = device

    @abstractmethod
    def forward(self, data):
        return self.model(data)

