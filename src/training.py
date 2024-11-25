from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.sttransformer import STTransformer
from src import create_datasets
from src.cbfocal_loss import FocalLoss
from src.dataset import print_target_distribution
from src.models.abstract_model import AbstractModel


def train(
        model,
        dataloader,
        optimizer,
        criterion,
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


def evaluate(
        model,
        dataloader,
        criterion,
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

    classes, counts = th.unique(all_preds, return_counts=True)
    pred_dist = th.zeros(3)
    pred_dist[classes] = counts / counts.sum()  # Are we abusing common class?

    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    return losses.sum(), losses.mean(), accuracies.mean(), f1_weighted, pred_dist


def train_model(
        x_dim: int,
        seq_len: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        # TODO only pass the model, not all the parameters
        # Begin model specific parameters
        d_model: int = 128,
        num_heads: int = 2,
        num_encoders: int = 8,
        num_outputs: int = 3,
        num_lstm_layers: int = 3,
        lstm_dim: int = 256,
        k: int = 64,
        fc_dim: int = 512,
        fc_dropout: float = 0.2,
        lr: float = 0.0002,
        time_idx: list[int] = None,
        # End model specific parameters
        # TODO These are top level training parameters, should be moved to a config
        optimizer: str = "adam",
        scheduler: str = "plateau",
        criterion: str = "ce",
        epochs: int = 20,
        train_label_ct: th.Tensor = None,
        model_class: AbstractModel = STTransformer,
        writer: SummaryWriter = None,
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
    # Set the optimizer
    match optimizer:
        case "adam":
            optimizer = th.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.99),
                eps=1e-8,
                weight_decay=1e-3,
            )
        case "sgd":
            optimizer = th.optim.SGD(model.parameters(), lr=lr)
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    # Set the scheduler
    match scheduler:
        case "plateau":
            scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5)
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
                gamma=2.0
            )
        case _:
            raise ValueError(f"Unknown criterion: {criterion}")

    train_losses = np.zeros(epochs)
    valid_losses = np.zeros(epochs)

    pb = tqdm(total=epochs, desc="Epochs")
    for epoch in range(epochs):
        # Run training over the batches
        train_loss, train_loss_avg = train(model, train_loader, optimizer, criterion, device, epoch, writer=writer)
        scheduler.step(train_loss)

        # Evaluate the validation set
        valid_loss, valid_loss_avg, _, _, v_pred_dist = evaluate(model, valid_loader, criterion, device=device)

        # Log the progress
        train_losses[epoch] = train_loss_avg
        valid_losses[epoch] = valid_loss_avg
        # TODO add checkpointing for model parameters, saving the best model, etc.
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss_avg, epoch)
            writer.add_scalar("Loss/valid", valid_loss_avg, epoch)

        # Update the progress bar to also show the loss
        pred_string = " - ".join([f"C{ix} {x:.3f}" for ix, x in enumerate(v_pred_dist)])
        pb.set_description(
            f"E: {epoch + 1} | Train: {train_loss_avg:.4f} | Valid: {valid_loss_avg:.4f} | V_Pred Dist: {pred_string}")
        pb.update(1)

    # Evaluate the test set
    test_loss, test_loss_avg, test_acc, test_f1, test_pred_dist = evaluate(
        model,
        test_loader,
        criterion,
        device=device
    )

    return train_losses, valid_losses, test_loss, test_loss_avg, test_acc, test_f1, test_pred_dist


def run_experiment(symbol: str, seq_len: int, batch_size: int, log_splits: bool = False, **kwargs):
    """
    Load the data symbol and create PriceSeriesDatasets.

    The symbol data is loaded into PriceSeriesDatasets, augmented with indicators
    and then partitioned into train, validation, and test sets. The data are then
    wrapped in DataLoader objects for use in training and evaluation loops.

    :param symbol: The symbol to load
    :param seq_len: The sequence length to use
    :param batch_size: The batch size to use for DataLoader
    :param log_splits: Whether to log the target distribution
    :param kwargs: Additional arguments for the model
    """
    train_data, valid_data, test_data = create_datasets(
        symbol,
        seq_len=seq_len,
        # fixed_scaling=[(7, 3000.), (8, 12.), (9, 31.)],
        log_splits=log_splits
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    train_label_ct = train_data.target_counts

    return train_model(
        x_dim=train_data[0][0].shape[1],
        seq_len=seq_len,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_label_ct=train_label_ct,
        **kwargs
    )


def run_grid_search(
        trial_prefix: str = "default_trial",
        use_writer: bool = True
):
    """
    WIP - Will be parameterizing to allow for grid search over a model

    Sorry, yes this is a dirty..dirty grid search.
    TODO - add model type, default config generator
    """
    th.manual_seed(1984)
    # TODO Convert this to a method that returns  configurations to iterate over
    # if args aren't passed, a default will be used so everything should be optional\
    # requires model type arg to add keys that are specific to the model
    # TODO step one - refactor the configurations to be passed in as args
    ctx_size = [30]
    d_models = [64]
    batch_sizes = [64]
    l_rates = [5e-6, 1e-5, 5e-5]
    fc_dims = [512]
    fc_dropouts = [0.1, 0.2, 0.3]
    mlp_dims = [512]
    mlp_dropouts = [0.2, 0.3, 0.4]
    n_freqs = [32]
    num_encoders = [2]
    num_heads = [2, 4, 8]
    num_lstm_layers = [2]
    lstm_dim = [128]
    criteria = ["ce"]

    # use itertools.product to generate dictionaries of hyperparameters
    configurations = [
        {
            "symbol": "atnf",
            "seq_len": ctx,
            "batch_size": bs,
            "d_model": d,
            "lr": lr,
            "time_idx": [0, 6, 7, 8],
            "fc_dim": fc,
            "fc_dropout": fcd,
            "mlp_dim": mlp,
            "mlp_dropout": mld,
            "k": k,
            "num_encoders": ne,
            "num_heads": nh,
            "num_lstm_layers": nl,
            "lstm_dim": ld,
            "optimizer": "adam",
            "scheduler": "plateau",
            "criterion": crit,  # Cross Entropy
            "epochs": 200
        }
        for d, lr, fc, fcd, mlp, mld, k, ne, nh, nl, ld, ctx, bs, crit in product(
            d_models,
            l_rates,
            fc_dims,
            fc_dropouts,
            mlp_dims,
            mlp_dropouts,
            n_freqs,
            num_encoders,
            num_heads,
            num_lstm_layers,
            lstm_dim,
            ctx_size,
            batch_sizes,
            criteria
        )
    ]

    # TODO refactor in a nicer way
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
        "test_loss": [],
        "test_loss_avg": [],
        "test_acc": [],
        "test_f1": [],
        "time_idx": [],
        "test_pred_dist": [],
        "mlp_dropout": [],
        "mlp_dim": []
    }

    for trial, config in enumerate(configurations):
        writer = SummaryWriter(log_dir=f"../data/tensorboard/{trial_prefix}_{trial:03}") if use_writer else None
        tr_loss, v_loss, tst_loss, tst_loss_avg, tst_acc, tst_f1, tst_pred_dist = (run_experiment
            (
            log_splits=trial == 0,
            writer=writer,
            **config
        ))

        print_target_distribution([("Test", tst_pred_dist)])

        plt.plot(tr_loss, label='Train Loss')
        plt.plot(v_loss, label='Valid Loss')
        # Set y scale between 0.25 and 1.25
        plt.xlim(0, config['epochs'])
        plt.ylim(0.0, 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"../figures/trial_{trial:03}_{config['symbol']}_loss.png")
        plt.close()

        # Save the loss and training results to the dictionary
        results_dict["trial_id"].append(trial)
        results_dict["test_loss"].append(tst_loss)
        results_dict["test_loss_avg"].append(tst_loss_avg)
        results_dict["test_acc"].append(tst_acc)
        results_dict["test_f1"].append(tst_f1)
        results_dict["test_pred_dist"].append([round(x.item(), 3) for x in tst_pred_dist])

        # Iterate over the config and append the values to the dictionary
        for key, value in config.items():
            results_dict[key].append(value)

    # Save the results to a CSV file
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f"../data/{trial_prefix}_results.csv", index=False)


if __name__ == '__main__':
    run_grid_search()
