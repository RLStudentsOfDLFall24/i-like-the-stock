from ncps.torch import CfC

from src.models.abstract_model import AbstractModel

<<<<<<< HEAD
class CfC_LNN(AbstractModel):
=======
class LNN_CfC(AbstractModel):
>>>>>>> 50185f2 (Working out a couple of changes)
    def __init__(self,
                 d_features,
                 hidden_size,
                 output_size,
                 backbone_dropout=0.0,
                 backbone_layers=1,
                 backbone_hidden=128,
                 activation='lecun_tanh',
                 use_mixed=False,
                 device='cpu',
                ):
<<<<<<< HEAD
        super(CfC_LNN, self).__init__(d_features=d_features, device=device)
=======
        super(LNN_CfC, self).__init__(d_features=d_features, device=device)
>>>>>>> 50185f2 (Working out a couple of changes)
        
        self.model = CfC(
            d_features,
            hidden_size,
            proj_size=output_size,
            backbone_dropout=backbone_dropout,
            backbone_layers=backbone_layers,
            backbone_units=backbone_hidden,
            activation=activation,
            return_sequences=False,
            mixed_memory=use_mixed).to(device)

    def forward(self, x):
        result = self.model(x)

        return result[0]
