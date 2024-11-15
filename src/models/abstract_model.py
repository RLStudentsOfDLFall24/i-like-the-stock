import torch
from torch import nn
from abc import ABC, ABCMeta, abstractmethod

#Inherit from both
class AbstractModelMeta(nn.Module, metaclass=ABCMeta):
    pass


#define a model template
class AbstractModel(AbstractModelMeta):
    model: nn.Module

    @abstractmethod
    def forward(self, data):
        return self.model(data)

