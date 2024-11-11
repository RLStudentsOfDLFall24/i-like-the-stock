import torch
from torch import nn
from abc import ABC, ABCMeta

#Inherit from both
class AbstractModelMeta(ABCMeta, nn.Module):
    pass


#define a model template
class AbstractModel(AbstractModelMeta):
    def forward(self, data):
        pass

