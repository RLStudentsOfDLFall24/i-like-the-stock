import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
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


# this source is derived from the paper's algorithm, this walkthrough: https://github.com/KPEKEP/LTCtutorial/blob/main/LNN_LTC_Tutorial_Eng.ipynb
# and the original source from the paper found here: https://github.com/raminmh/liquid_time_constant_networks
class LNN(AbstractModel):
    def __init__(self, d_features, hidden_size, output_size, n_layers=6, activation='relu', eps=1e-8, device='cpu', is_affine=True):
        super(LNN, self).__init__(d_features=d_features)
        self.step_size = 1 / n_layers

        self.activation = get_activation_function(activation)

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.epsilon = eps
        self.device = device

        # pulled this setup from the NCPS version of the LTC implemenation. I want to know what everything does!
        # https://github.com/mlech26l/ncps/blob/master/ncps/torch/ltc_cell.py#L57-L67
        self._param_configs = {
            # leak conductance
            "gleak": {'ranges': (0.001, 1.0), 'requires_grad': True, 'shape': (self.hidden_size,)},
            # leak voltage
            "vleak": {'ranges': (-0.2, 0.2), 'requires_grad': True, 'shape': (self.hidden_size,)},
            # membrane capacitance, also is the result of 1/Tau, where Tau is the time constant
            "cm": {'ranges': (0.4, 0.6), 'requires_grad': True, 'shape': (self.hidden_size,)},
            "w": {'ranges': (0.001, 1.0), 'requires_grad': True, 'shape': (self.hidden_size, self.hidden_size)}, # Weight
            "sigma": {'ranges': (3, 8), 'requires_grad': True, 'shape': (self.hidden_size, self.hidden_size)}, # Scale
            "mu": {'ranges': (0.3, 0.8), 'requires_grad': True, 'shape': (self.hidden_size, self.hidden_size)}, # Mean
            # Reversal Potentionals for neuron connections
            "erev": {'ranges': (-0.2, 0.2), 'requires_grad': True, 'shape': (self.hidden_size, self.hidden_size)},
            # adjacency matrix for connections between neurons
            'sparsity_mask': {'ranges': (0, 1), 'requires_grad': False, 'shape': (self.hidden_size, self.hidden_size)},
            "sensory_w": {'ranges': (0.001, 1.0), 'requires_grad': True, 'shape': (self.d_features, self.hidden_size)}, # Weight
            "sensory_sigma": {'ranges': (3, 8), 'requires_grad': True, 'shape': (self.d_features, self.hidden_size)}, # Scale
            "sensory_mu": {'ranges': (0.3, 0.8), 'requires_grad': True, 'shape': (self.d_features, self.hidden_size)}, # Mean
            # Reversal Potentionals for sensory inputs between neurons
            "sensory_erev": {'ranges': (-0.2, 0.2), 'requires_grad': True, 'shape': (self.d_features, self.hidden_size)},
            # adjacency matrix for the sensory inputs between neurons
            'sensory_sparsity_mask': {'ranges': (0, 1), 'requires_grad': False, 'shape': (self.d_features, self.hidden_size)},
        }

        self.__allocate_parameters()
        self.output = torch.nn.Linear(hidden_size, output_size, bias=is_affine, device=device)

    def __allocate_parameters(self):
        self._params = {}

        for name in self._param_configs:
            config = self._param_configs[name]

            minval, maxval = config['ranges']

            # loosely derived from this source: https://github.com/mlech26l/ncps/blob/master/ncps/torch/ltc_cell.py#L105-L110
            if minval == maxval:
                init_values = torch.ones(config['shape'], dtype=torch.float32) * minval
            else:
                init_values = torch.rand(*config['shape'], dtype=torch.float32) * (maxval - minval) + minval

            param = torch.nn.Parameter(init_values.to(self.device), requires_grad=config['requires_grad'])
            self.register_parameter(name, param)

            self._params[name] = param

    def forward(self, data: torch.Tensor):
        with torch.enable_grad():
            N, T, _ = data.size()

            states = torch.zeros((N, self.hidden_size)).to(self.device)
            outputs: list[torch.Tensor] = []
            for idx in range(T):
                output, states = self.__ode_solver(data[:, idx, :], states)
                outputs.append(output)
            
            return outputs[-1]

    def __activate(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1) # for broadcasting
        activation = sigma * (v_pre - mu)
        return self.activation(activation)
    
    def __ode_solver(self, data: torch.Tensor, hidden: torch.Tensor, elapsed_time=1.0):
        v_pre = hidden

        # Pre-compute the effects of the sensory neurons
        sensory_activation = F.softplus(self._params['sensory_w']) * self.__activate(data, self._params['sensory_mu'], self._params['sensory_sigma'])
        sensory_activation += self._params['sensory_sparsity_mask']
        sensory_reversal_activation = sensory_activation * self._params['sensory_erev']

        # calculate the sensory input numerator and denominator
        w_numerator_sensory = torch.sum(sensory_reversal_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_activation, dim=1)
        
        # calculate membrane capacitance over time
        cm_t = F.softplus(self._params['cm']) / (elapsed_time / self.n_layers) # CM / delta T

        # initialize weights for neuron connections
        w_param = F.softplus(self._params['w'])
        for _ in range(self.n_layers):
            v_pre = self.__fused_step(v_pre, w_param, w_numerator_sensory, w_denominator_sensory, cm_t)

        outputs = self.output(v_pre)
        
        return outputs, v_pre

    def __fused_step(self, 
                     v_pre: torch.Tensor, 
                     w_param: torch.nn.Parameter, 
                     w_numerator_sensory: torch.Tensor, 
                     w_denominator_sendsory: torch.Tensor, 
                     cm_t: torch.Tensor):
        # Activation based on previous state
        w_activation = w_param * self.__activate(v_pre, self._params['mu'], self._params['sigma'])
        w_activation *= self._params['sparsity_mask']
        reversal_activation = w_activation + self._params['erev']

        # calculate the numerator and denominator for the neuron connections
        w_numerator = torch.sum(reversal_activation, dim=1) + w_numerator_sensory
        w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sendsory

        # Leak conductance and voltage calculations, and updates the state (voltage)
        gleak = F.softplus(self._params['gleak'])

        return (cm_t * v_pre + gleak * self._params['vleak'] + w_numerator) / (cm_t + gleak + w_denominator + self.epsilon)


from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP 

class LNN_2(AbstractModel):
    def __init__(self, d_features, hidden_size, output_size, n_layers=6, use_mixed=False):
        super(LNN_2, self).__init__(d_features=d_features)
        wiring = AutoNCP(d_features, d_features)
        
        self.model = LTC(d_features, wiring, ode_unfolds=n_layers, mixed_memory=use_mixed, return_sequences=False)

    def forward(self, x):
        result = self.model(x)

        return result[0]

class LNN_CfC(AbstractModel):
    def __init__(self,
                 d_features,
                 hidden_size,
                 output_size,
                 backbone_dropout=0.0,
                 backbone_layers=1,
                 backbone_hidden=128,
                 activation='lecun_tanh',
                 use_mixed=False):
        super(LNN_CfC, self).__init__(d_features=d_features)
        wiring = AutoNCP(hidden_size, output_size)
        
        self.model = CfC(
            d_features,
            hidden_size,
            proj_size=output_size,
            backbone_dropout=backbone_dropout,
            backbone_layers=backbone_layers,
            backbone_units=backbone_hidden,
            activation=activation,
            return_sequences=False,
            mixed_memory=use_mixed)

    def forward(self, x):
        result = self.model(x)

        return result[0]
