import torch
from torch import nn
from abc import ABC, ABCMeta, abstractmethod

#Inherit from both
class AbstractModelMeta(nn.Module, metaclass=ABCMeta):
    pass


#define a model template
class AbstractModel(AbstractModelMeta):
    model: nn.Module

    def __init__(self, d_features: int):
        super().__init__()
        self.d_features = d_features

    @abstractmethod
    def forward(self, data):
        return self.model(data)

