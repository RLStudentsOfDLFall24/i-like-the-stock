import torch
from torch import nn
from .abstract_model import AbstractModel


def get_activation_function(activation):
    match activation:
        case 'relu':
            return nn.ReLU()
        case 'leaky_relu':
            return nn.LeakyReLU()
        case 'tanh':
            return nn.Tanh()
        case 'sigmoid':
            return nn.Sigmoid()
        case 'elu':
            return nn.ELU()
        case 'gelu':
            return nn.GELU()
        case _:
            raise Exception(f'{activation} is not currently supported')


class LNN(AbstractModel):
    def __init__(self, batch_size, input_size, hidden_size, output_size, n_layers=6, activation='relu'):
        super(LNN, self).__init__(batch_size=batch_size)
        self.step_size = 1 / n_layers

        self.activation = get_activation_function(activation)

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.time_constant = nn.Parameter(torch.ones(hidden_size))
        self.weight = nn.Parameter(torch.randn(input_size, hidden_size))
        self.r_weight = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.A = nn.Parameter(torch.ones(hidden_size))

        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.r_weight.data)

        self.output = nn.Linear(hidden_size, output_size)
        

    def forward(self, data: torch.Tensor):
        with torch.enable_grad():
            N, T, _ = data.size()
            x = torch.zeros((N, self.hidden_size)).to(self.bias.device)
            for idx in range(T):
                for _ in range(self.n_layers):
                    x = self.__fused_step(data[:, idx, :], x)
            
            return self.output(x)
    
    def __fused_step(self, data: torch.Tensor, hidden: torch.Tensor):
        func_data = data @ self.weight + hidden @ self.r_weight + self.bias.unsqueeze(0)

        activation = self.activation(func_data)

        return (hidden + self.step_size * activation * self.A) / (1 + self.step_size * (1 / self.time_constant) + activation)


from ncps.torch import LTC
from ncps.wirings import AutoNCP 

class LNN_2(AbstractModel):
    def __init__(self, batch_size, input_size, hidden_size, output_size, n_layers=6, use_mixed=False):
        super(LNN_2, self).__init__(batch_size=batch_size)
        wiring = AutoNCP(hidden_size, output_size)
        
        self.model = LTC(input_size, wiring, ode_unfolds=n_layers, mixed_memory=use_mixed, return_sequences=False)

    def forward(self, x):
        result = self.model(x)

        return result[0]
