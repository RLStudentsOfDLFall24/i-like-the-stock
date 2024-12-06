from abstract_model import AbstractModel
import torch

class UniformRandomModel(AbstractModel):
    def __init__(self, **kwargs):
        super(UniformRandomModel, self).__init__(**kwargs)

    def forward(self, x: torch.Tensor):
        ''''''