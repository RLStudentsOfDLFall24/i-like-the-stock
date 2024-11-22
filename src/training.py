from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.sttransformer import STTransformer
from src import create_datasets
from src.cbfocal_loss import FocalLoss
from src.dataset import print_target_distribution
from src.models.abstract_model import AbstractModel


def train(model, dataloader, optimizer, criterion, device: th.device) -> tuple[float, float]:
    """
    Train the model on the given dataloader and return the losses for each batch.

    :param model:
    :param dataloader:
    :param optimizer:
    :param criterion:
    :param device:
    :return:
    """
    # Set the model to training mode
    model.train()

    # Initialize the loss
    # total_loss = 0.0
    losses = np.zeros(len(dataloader))
    # accuracies = np.zeros(len(dataloader))

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

        # Update loss
        # total_loss += loss.item()
        losses[ix] = loss.item()
        # accuracies[ix] = th.sum(th.argmax(logits, dim=1) == y).item() / y.shape[0]
        # pb.set_description_str("Batch: %d, Loss: %.4f" % ((ix + 1), loss.item()))

    return losses.sum(), losses.mean()


def evaluate(
        model,
        dataloader,
        criterion,
        print_dist: bool = False,
        device: th.device = 'cpu',
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

    # Log the prediction distribution over labels
    if print_dist:
        classes, counts = th.unique(all_preds, return_counts=True)
        pred_dist = th.zeros(3)
        pred_dist[classes] = counts / counts.sum() # Are we abusing common class?
        print_target_distribution([("Preds", pred_dist)])
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    return losses.sum(), losses.mean(), accuracies.mean(), f1_weighted, pred_dist if print_dist else None


def train_sttransformer(
        x_dim: int,
        seq_len: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        d_model: int = 128,
        num_heads: int = 2,
        num_encoders: int = 8,
        num_outputs: int = 3,
        num_lstm_layers: int = 3,
        lstm_dim: int = 256,
        k: int = 64,
        fc_dim: int = 512,
        fc_dropout: float = 0.2,
        lr: float = 0.00002,
        optimizer: str = "adam",
        scheduler: str = "plateau",
        criterion: str = "ce",
        epochs: int = 20,
        time_idx: list[int] = None,
        train_label_ct: th.Tensor = None,
        model_class: AbstractModel = STTransformer,
        **kwargs
):
    """Train a model and test the methods"""
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    model = STTransformer(
        d_features=x_dim,
        device=device,
        num_encoders=num_encoders,
        num_outputs=num_outputs,
        num_lstm_layers=num_lstm_layers,
        lstm_dim=lstm_dim,
        time_idx=[0] if time_idx is None else time_idx,
        model_dim=d_model,
        num_heads=num_heads,
        n_frequencies=k,
        fc_dim=fc_dim,
        fc_dropout=fc_dropout,
        ctx_window=seq_len
    )

    # EPOCHS = 20

    # Set the optimizer
    match optimizer:
        case "adam":
            optimizer = th.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8)
        case "sgd":
            optimizer = th.optim.SGD(model.parameters(), lr=lr)
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    # Set the scheduler
    match scheduler:
        case "plateau":
            scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        case "step":
            scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        case "multi":
            scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15])
        case _:
            raise ValueError(f"Unknown scheduler: {scheduler}")

    match criterion:
        case "ce":
            weight = None
            if train_label_ct is not None:
                weight = train_label_ct.max() / train_label_ct
                weight = weight / weight.sum()
                weight = weight.to(device)
            criterion = th.nn.CrossEntropyLoss(
                weight=weight
            )
        case "cb_focal":
            criterion = FocalLoss(
                class_counts=train_label_ct.to(device),
                gamma=1.0
            )
        case _:
            raise ValueError(f"Unknown criterion: {criterion}")

    train_losses = np.zeros(epochs)
    valid_losses = np.zeros(epochs)

    pb = tqdm(total=epochs, desc="Epochs")
    for epoch in range(epochs):
        # Run training over the batches
        train_loss, train_loss_avg = train(model, train_loader, optimizer, criterion, device)
        scheduler.step(train_loss)

        # Evaluate the validation set
        valid_loss, valid_loss_avg, _, _, _ = evaluate(model, valid_loader, criterion, print_dist=True, device=device)

        # Log the progress
        train_losses[epoch] = train_loss_avg
        valid_losses[epoch] = valid_loss_avg
        # TODO add checkpointing
        # Update the progress bar to also show the loss
        pb.set_description(f"Epoch: {epoch + 1} | Train: {train_loss_avg:.4f} | Valid: {valid_loss_avg:.4f}")
        pb.update(1)

    # Evaluate the test set
    test_loss, test_loss_avg, test_acc, test_f1, test_pred_dist = evaluate(
        model,
        test_loader,
        criterion,
        print_dist=True,
        device=device
    )

    return train_losses, valid_losses, test_loss, test_loss_avg, test_acc, test_f1, test_pred_dist


def run_experiment(symbol: str, seq_len: int, batch_size: int, **kwargs):
    train_data, valid_data, test_data = create_datasets(
        symbol,
        seq_len=seq_len,
        fixed_scaling = [(7, 3000.), (8, 12.), (9, 31.)],
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    train_label_ct = train_data.target_counts

    return train_sttransformer(
        x_dim=train_data[0][0].shape[1],
        seq_len=seq_len,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_label_ct=train_label_ct,
        **kwargs
    )


def run():
    th.manual_seed(1984)
    # We want to run a simple search over the hyperparameters
    ctx_size = [30]
    d_models = [64]
    batch_sizes = [64]
    l_rates = [5e-6]
    fc_dims = [1024]
    fc_dropouts = [0.3]
    n_freqs = [4, 8, 16]
    num_encoders = [4]
    num_heads = [4]
    num_lstm_layers = [2]
    lstm_dim = [128]

    # d_models = [64]
    # l_rates = [1e-6, 1e-5]
    # fc_dims = [2048]
    # fc_dropouts = [0.4]
    # n_freqs = [32]
    # num_encoders = [2]
    # num_heads = [2]
    # num_lstm_layers = [2]
    # lstm_dim = [256]

    # use itertools.product to generate dictionaries of hyperparameters
    configurations = [
        {
            "symbol": "atnf",
            "seq_len": ctx,
            "batch_size": 64,
            "d_model": d,
            "lr": lr,
            "time_idx": [0, 6, 7, 8],
            "fc_dim": fc,
            "fc_dropout": fcd,
            "k": k,
            "num_encoders": ne,
            "num_heads": nh,
            "num_lstm_layers": nl,
            "lstm_dim": ld,
            "optimizer": "adam",
            "scheduler": "plateau",
            # "criterion": "ce",
            "criterion": "cb_focal",
            "epochs": 100
        }
        for d, lr, fc, fcd, k, ne, nh, nl, ld, ctx, bs in product(
            d_models,
            l_rates,
            fc_dims,
            fc_dropouts,
            n_freqs,
            num_encoders,
            num_heads,
            num_lstm_layers,
            lstm_dim,
            ctx_size,
            batch_sizes
        )
    ]

    # create a dict that has lists for each hyperparameter
    results_dict = {
        "trial_id": [],
        "symbol": [],
        "seq_len": [],
        "batch_size": [],
        "d_model": [],
        "lr": [],
        "fc_dim": [],
        "fc_dropout": [],
        "k": [],
        "num_encoders": [],
        "num_heads": [],
        "num_lstm_layers": [],
        "lstm_dim": [],
        "optimizer": [],
        "scheduler": [],
        "criterion": [],
        "epochs": [],
        # "train_losses": [],
        # "valid_losses": [],
        "test_loss": [],
        "test_loss_avg": [],
        "test_acc": [],
        "test_f1": [],
        "time_idx": [],
        "test_pred_dist": []
    }

    for trial, config in enumerate(configurations):
        train_losses, v_losses, test_loss, test_loss_avg, test_acc, f1, test_pred_dist = run_experiment(**config)
        # abstract this up level
        # plot train and valid losses on the same graph
        plt.plot(train_losses, label='Train Loss')
        plt.plot(v_losses, label='Valid Loss')
        # Set y scale between 0.25 and 1.25
        plt.xlim(0, config['epochs'])
        plt.ylim(0.0, 1.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"../figures/trial_{trial}_{config['symbol']}_loss.png")
        plt.close()

        # Save the loss and training results to the dictionary
        results_dict["trial_id"].append(trial)
        # results_dict["train_losses"].append(train_losses)
        # results_dict["valid_losses"].append(v_losses)
        results_dict["test_loss"].append(test_loss)
        results_dict["test_loss_avg"].append(test_loss_avg)
        results_dict["test_acc"].append(test_acc)
        results_dict["test_f1"].append(f1)
        results_dict["test_pred_dist"].append([round(x.item(), 3) for x in test_pred_dist])

        # Iterate over the config and append the values to the dictionary
        for key, value in config.items():
            results_dict[key].append(value)

    # Save the results to a CSV file
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv("../data/st_trial_results.csv", index=False)


if __name__ == '__main__':
    run()
