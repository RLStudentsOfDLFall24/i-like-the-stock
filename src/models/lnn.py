import torch
from torch import nn
from .abstract_model import AbstractModel


class LNN(AbstractModel):
    def __init__(self, batch_size, input_size, hidden_size, n_layers=6, activation=nn.ReLU):
        super(LNN, self).__init__(batch_size=batch_size)
        self.step_size = 1 / n_layers
        self.activation = activation
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layers = n_layers

        self.time_constant = nn.Parameter(torch.ones(hidden_size))
        self.weight = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.r_weight = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.A = nn.Parameter(torch.ones(hidden_size))

        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.r_weight.data)
        

    def forward(self, data: torch.Tensor):
        N, T, D = data.size()
        x_t_n = torch.zeros(self.hidden_size)
        return super().forward(data)
    
    def __fused_step(self, data: torch.Tensor, hidden: torch.Tensor):
        func_data = self.weight @ data + self.r_weight @ hidden + self.bias

        activation = self.activation(func_data)

        return (hidden + self.step_size * activation * self.A) / (1 + self.step_size * (1 / self.time_constant) + activation)
