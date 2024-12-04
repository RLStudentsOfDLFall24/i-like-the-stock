import torch as th
import numpy as np

from sklearn.metrics import f1_score, matthews_corrcoef
from src.cbfocal_loss import FocalLoss
from src.models.abstract_model import AbstractModel
from torch.utils.data import DataLoader

from src.simulation._simulation import simulate_trades

def evaluate(
        model: AbstractModel,
        dataloader: DataLoader,
        criterion: th.nn.CrossEntropyLoss | FocalLoss,
        device: th.device = 'cpu',
        simulate: bool = False,
) -> tuple[float, float, float, float, th.Tensor | None]:
    # Set the model to evaluation
    model.eval()
    # total_loss = 0.0

    losses = np.zeros(len(dataloader))
    accuracies = np.zeros(len(dataloader))

    all_preds = []
    all_labels = []
    with th.no_grad():
        for ix, data in enumerate(dataloader):
            x = data[0].to(device)
            y = data[1].to(device)

            logits = model(x)
            # Get the argmax of the logits
            preds = th.argmax(logits, dim=1)
            all_preds.append(preds)
            all_labels.append(y)

            losses[ix] = criterion(logits, y).item()
            accuracies[ix] = th.sum(th.argmax(logits, dim=1) == y).item() / y.shape[0]

    # concat the preds and labels, then send to cpu
    all_preds = th.cat(all_preds).cpu()
    all_labels = th.cat(all_labels).cpu()

    # We can use the simulation code to produce a results figure
    data = dataloader.dataset
    if simulate:
        results_df = simulate_trades(
            data.unscaled_prices.detach().cpu().numpy(),
            all_preds.numpy(),
            data.time_idx.detach().cpu().numpy()
        )

    classes, counts = th.unique(all_preds, return_counts=True)
    pred_dist = th.zeros(3)
    pred_dist[classes] = counts / counts.sum()  # Are we abusing common class?

    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_preds)

    return losses.sum(), losses.mean(), accuracies.mean(), f1_weighted, pred_dist, mcc, results_df if simulate else None
