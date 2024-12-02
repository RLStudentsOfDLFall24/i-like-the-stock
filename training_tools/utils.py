from typing import Optional
from src.cbfocal_loss import FocalLoss
from src.models.abstract_model import AbstractModel

import torch
import matplotlib.pyplot as plt

def get_optimizer(type: str, model: AbstractModel, lr, **kwargs):
  match type:
    case 'adam':
      return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        **kwargs,
      )
    case 'sgd':
      return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        **kwargs,
      )
    case _:
      raise ValueError(f'Unknown optimizer: {type}')
  

def get_scheduler(type: str, optimizer: torch.optim.Optimizer, **kwargs):
  match type:
    case 'plateau':
      return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    case 'step':
      return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    case 'multi':
      return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    case _:
      raise ValueError(f'Unknown scheduler: {type}')
    

def get_criterion(type: str, train_label_ct: Optional[torch.Tensor] = None, device='cpu', **kwargs):
  match type:
    case 'ce':
      weight = None
      if train_label_ct is not None:
        weight = train_label_ct.max() / train_label_ct
        weight = weight / weight.sum()
        weight = weight.to(device)

      return torch.nn.CrossEntropyLoss(
        weight=weight,
      )
    case 'cb_focal':
      return FocalLoss(
        class_counts=train_label_ct.to(device),
        **kwargs,
      )


def plot_results(tr_loss, v_loss, epochs, image_name = None):
    plt.plot(tr_loss, label='Train Loss')
    plt.plot(v_loss, label='Valid Loss')
    # Set y scale between 0.25 and 1.25
    plt.xlim(0, epochs)
    plt.ylim(0.0, 2)
    plt.legend()
    plt.tight_layout()
    if image_name is not None:
        plt.savefig(f"../figures/{image_name}.png")
    else:
      plt.show()
    plt.close()
