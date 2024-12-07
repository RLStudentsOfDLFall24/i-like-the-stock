import torch
from torch import nn
from src.models.abstract_model import AbstractModel


class RNN(AbstractModel):
    def __init__(self,
                 d_features: int,
                 num_outputs: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 init_scaling: float,
                 device: torch.device):
        super(RNN, self).__init__(d_features=d_features, device=device)

        self.model = nn.RNN(input_size=d_features,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=num_layers,
                            dropout=dropout)

        #<IRNN paper implementation>
        self.model.weight_hh_l0 = torch.nn.Parameter(torch.nn.init.eye_(self.model.weight_hh_l0)*init_scaling)
        torch.nn.init.zeros_(self.model.bias_hh_l0)
        self.model.weight_ih_l0 = torch.nn.Parameter(torch.nn.init.eye_(self.model.weight_ih_l0)*init_scaling)
        #torch.nn.init.eye_(self.model.weight_ih_l0)
        torch.nn.init.zeros_(self.model.bias_ih_l0)
        #</IRNN paper>

        self.fc_output = nn.Linear(hidden_size, num_outputs)

        self.to(device)

    def forward(self, data):
        fwd = super().forward(data)
        #hidden = fwd[1]
        out = self.fc_output(fwd[0])
        return out[:,-1,:]

