import torch as th
import numpy as np

from src.cbfocal_loss import FocalLoss
from src.models.abstract_model import AbstractModel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def train(
        model: AbstractModel,
        dataloader: DataLoader,
        optimizer: th.optim.Optimizer,
        criterion: th.nn.CrossEntropyLoss | FocalLoss,
        device: th.device,
        epoch,
        writer: SummaryWriter = None
) -> tuple[float, float]:
    """
    Train the model on the given dataloader and return the losses for each batch.

    :param model:
    :param dataloader:
    :param optimizer:
    :param criterion:
    :param device:
    :param epoch:
    :param writer:
    :return:
    """
    model.train()

    # Initialize the loss
    losses = np.zeros(len(dataloader))

    for ix, data in enumerate(dataloader):
        x = data[0].to(device)
        y = data[1].to(device)

        optimizer.zero_grad()
        logits = model(x)
        # If we only have one output, we need to unsqueeze the y tensor
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)

        loss = criterion(logits, y)

        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if writer is not None:
            # Log gradients at the end of the epoch
            for l, (name, param) in enumerate(model.named_parameters()):
                if param.grad is not None:
                    writer.add_scalar(f"Gradients/{l:02}_{name}", param.grad.norm().item(),
                                      epoch * len(dataloader) + ix)

        # Update loss
        losses[ix] = loss.item()

    return losses.sum(), losses.mean()
