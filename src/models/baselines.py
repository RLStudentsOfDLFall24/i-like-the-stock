from .abstract_model import AbstractModel
import torch
import numpy as np


class RiddlerModel(AbstractModel):
    '''
    This model makes probabilistic decisions
    around buys, sells, and holding. This way
    we can use our intuitions about what the best
    rate should be, and test our models against
    that baseline.
    '''
    def __init__(self, d_features: int, device: torch.device, buy_percent=0.25, sell_percent=0.70, hold_percent=0.05, **kwargs):
        super(RiddlerModel, self).__init__(d_features, device)

        self.rng = np.random.default_rng(kwargs['seed'] if 'seed' in kwargs else None)

        self.these_parameters_do_nothing = torch.nn.Parameter(torch.ones((84, 84)).to(device), requires_grad=True)

        self.percents = torch.Tensor([buy_percent, hold_percent, sell_percent])
        
        if self.percents.sum() != 1.0:
            self.percents = torch.nn.functional.softmax(self.percents, dim=-1)

    def forward(self, x: torch.Tensor):
        N, _, _ = x.shape

        output = torch.ones((N, 3)) * .42

        rand_ind = self.rng.choice(3, N, p=self.percents.numpy())
        output[torch.arange(N), rand_ind] *= 4

        return output.to(self.device)
