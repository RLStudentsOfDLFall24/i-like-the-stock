import os
from typing import Optional
from src.cbfocal_loss import FocalLoss
from src.dataset._dataset_utils import create_datasets
from src.dataset._priceseriesdataset import PriceSeriesDataset
from src.models.abstract_model import AbstractModel
from torch.utils.data import DataLoader, ConcatDataset

import torch
import matplotlib.pyplot as plt

def get_optimizer(type: str, model: AbstractModel, lr, config: dict):
  match type:
    case 'adam':
      betas = tuple(config['betas']) if 'betas' in config else (0.9, 0.999)
      config['betas'] = betas

      return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        **config,
      )
    case 'sgd':
      return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        **config,
      )
    case _:
      raise ValueError(f'Unknown optimizer: {type}')
  

def get_scheduler(type: str, optimizer: torch.optim.Optimizer, config: dict):
  match type:
    case "plateau":
        min_lr = config['min_lr'] if 'min_lr' in config else 1e-5
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=min_lr)
    case "step":
        step_size = config['step_size'] if 'step_size' in config else 10
        gamma = config['gamma'] if 'gamma' in config else 0.1
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    case "multi":
        milestones = config['milestones'] if 'milestones' in config else [5, 10, 15]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
    case _:
        raise ValueError(f"Unknown scheduler: {type}")


def get_criterion(type: str, train_label_ct: Optional[torch.Tensor] = None, trainer_params = None, device='cpu'):
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
        gamma=trainer_params['cbf_gamma'],
        beta=trainer_params['cbf_beta'],
      )


def get_data(
      train_symbols: list[str],
      batch_size: int,
      seq_len: int,
      target_symbol: str,
      log_splits: bool = False,
      root: str = '.',
    ) -> tuple[PriceSeriesDataset, torch.Tensor, DataLoader, DataLoader, DataLoader]:
    trains = []
    target_train = None
    target_valid = None
    target_test = None

    for symbol in train_symbols:
        train_data, valid_data, test_data = create_datasets(
            symbol,
            seq_len=seq_len,
            fixed_scaling=[(7, 3000.), (8, 12.), (9, 31.)],
            log_splits=log_splits,
            root=f"{root}/data/clean"
        )

        trains.append(train_data)
        if symbol == target_symbol:
            target_train = train_data
            target_valid = valid_data
            target_test = test_data

    concatted_trains = ConcatDataset(trains)
    train_label_ct = torch.sum(torch.stack([x.target_counts for x in trains]),dim=0)

    train_loader = DataLoader(concatted_trains, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(target_valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(target_test, batch_size=batch_size, shuffle=False)

    return target_train, train_label_ct, train_loader, valid_loader, test_loader


def plot_results(tr_loss, v_loss, epochs, y_lims=(0.0, 2.0), root = '.', image_name = None):
    # Create the figures directory if it doesn't exist
    if not os.path.exists(f"{root}/figures"):
        print("Creating figures directory...")
        os.makedirs(f"{root}/figures")

    plt.plot(tr_loss, label='Train Loss')
    plt.plot(v_loss, label='Valid Loss')
    # Set y scale between 0.25 and 1.25
    plt.xlim(0, epochs)
    plt.ylim(*y_lims)
    plt.legend()
    plt.tight_layout()
    if image_name is not None:
        plt.savefig(f"../figures/{image_name}.png")
    else:
      plt.show()
    plt.close()
