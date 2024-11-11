import torch
from torch import nn
from abc import ABC, ABCMeta, abstractmethod

#Inherit from both
class AbstractModelMeta(nn.Module, metaclass=ABCMeta):
    pass


#define a model template
class AbstractModel(AbstractModelMeta):
    @abstractmethod
    def forward(self, data):
        pass

