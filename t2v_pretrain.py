from datetime import datetime
import os
from itertools import product
from json import dumps

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import NextRowDataset
from src.models.custom_modules import T2V, STEmbedding
from src.models import T2VPretrainer


def get_daterange_data(
        start_date: str = "1900-01-01",
        end_date: str = "2100-01-01",
        file_path: str = "data/date_pretraining.csv",
        device: th.device = "cpu"
) -> th.Tensor:
    """
    Generate a range of dates and convert to yyyy, mm, dd then drop to disk.

    If the file already exists, read from disk and return the data. Otherwise,
    we generate a range of dates using pandas functionality to ease the process.
    The dates are then stored in column format with year, month, and day. The
    year, month and day are normalized to the range [0, 1] by using the following
    scaling factors:
    - year: 3000
    - month: 12
    - day: 31

    :param start_date: The start date for the range of dates.
    :param end_date: The end date for the range of dates.
    :param file_path: The file path to store the data.
    :param device: The device to store the data on.
    :return: A tensor with year, month, and day normalized values.
    """
    if os.path.exists(file_path):
        dates_df = pd.read_csv(file_path)
        return th.Tensor(dates_df.values).to(device)

    dates = pd.date_range(start=start_date, end=end_date)
    dates_df = pd.DataFrame(dates, columns=["date"])
    dates_df["year"] = dates_df["date"].dt.year / 3000.0
    dates_df["month"] = dates_df["date"].dt.month / 12.0
    dates_df["day"] = dates_df["date"].dt.day / 31.0

    # Day of the week
    # dates_df["dow"] = dates_df["date"].dt.dayofweek / 6.0
    dates_df.drop(columns=["date"], inplace=True)
    dates_df.to_csv("data/date_pretraining.csv", index=False)

    return th.Tensor(dates_df.values).to(device)


def split_next_row_data(
        data: th.Tensor, train_pct: float
) -> tuple[NextRowDataset, NextRowDataset]:
    """
    Shuffle and split the data into training and validation sets.

    :param data: The data to split.
    :param train_pct: The pct of the training set.
    :return: The training and validation sets.
    """
    # Make targets
    x, y = data[:-1], data[1:]

    permute = th.randperm(x.shape[0])
    x, y = x[permute], y[permute]

    train_size = int(train_pct * data.shape[0])
    x_train, x_valid = x[:train_size], x[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]

    return NextRowDataset(x_train, y_train), NextRowDataset(x_valid, y_valid)


def train_model(
        model: th.nn.Module,
        data_loader: DataLoader,
        optimizer: th.optim.Optimizer,
        criterion: th.nn.Module,
        device: th.device,
        writer: SummaryWriter,
        epoch: int,
) -> float:
    model.train()
    losses = np.zeros(len(data_loader))

    for ix, data in enumerate(data_loader):
        x, y = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        preds = model(x)

        loss = criterion(preds, y)

        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses[ix] = loss.item()
        for l, (name, param) in enumerate(model.named_parameters()):
            if param.grad is not None:
                writer.add_scalar(f"grads/{l:02}_{name}", param.grad.norm().item(),
                                  epoch * len(data_loader) + ix)
    return losses.mean()


def evaluate(
        model: th.nn.Module,
        data_loader: DataLoader,
        criterion: th.nn.Module,
        device: th.device,
) -> float:
    """
    Simple evaluation loop for the pretraining task.
    """
    # Set to evaluation mode
    model.eval()
    losses = np.zeros(len(data_loader))

    with th.no_grad():
        for ix, data in enumerate(data_loader):
            x, y = data[0].to(device), data[1].to(device)
            preds = model(x)
            loss = criterion(preds, y)
            losses[ix] = loss.item()

    return losses.mean()


def train_t2v(
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: th.device,
        epochs: int = 100,
        lr: float = 1e-3,
        writer_dir: str = "data/tensorboard",
        trial_name: str = "t2v_pretraining",
        **kwargs
) -> tuple[float, dict]:
    # Setup log writer and model
    writer = SummaryWriter(f"{writer_dir}/{trial_name}")
    model = T2VPretrainer(device=device, **kwargs)

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = th.nn.MSELoss()
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    pb = tqdm(range(epochs), desc="Epochs")
    best_loss = th.inf
    best_t2v_state = None
    for epoch in range(epochs):
        # Train on the training data
        train_loss_mean = train_model(
            model, train_loader, optimizer, criterion, device, writer, epoch
        )

        # Evaluate on the validation data
        valid_loss_mean = evaluate(
            model, val_loader, criterion, device
        )

        writer.add_scalar("loss/train", train_loss_mean, epoch)
        writer.add_scalar("loss/valid", valid_loss_mean, epoch)

        # Is this the best so far?
        if valid_loss_mean < best_loss:
            best_loss = valid_loss_mean
            best_t2v_state = model.t2v.state_dict()

        # Update the learning rate scheduler
        scheduler.step(valid_loss_mean)

        # Update the progress bar
        pb.set_description(
            f"E: {epoch + 1} | train: {train_loss_mean:.6f} | valid: {valid_loss_mean:.6f} | best: {best_loss:.6f}")
        pb.update(1)

    # Save the best model from this run
    best_t2v_state = best_t2v_state if best_t2v_state is not None else model.t2v.state_dict()
    return best_loss, best_t2v_state


def run_pretraining():
    run_start_ts = int(datetime.now().timestamp())
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    th.manual_seed(42)

    date_range_data = get_daterange_data(device=device)

    # Shuffle and split the data into training and validation sets
    train_data, val_data = split_next_row_data(date_range_data, 0.85)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=True)

    lrs = np.logspace(-6, -3, 6)
    mlp_dims = [1024, 512]
    frequencies = [64]

    model_param_combos = [
        {
            "input_dim": 3,
            "n_frequencies": n_freq,
            "mlp_dim": mlp_dim,
            "lr": lr,
        }
        for lr, mlp_dim, n_freq in product(lrs, mlp_dims, frequencies)
    ]

    best_loss, best_trial_name, best_params = th.inf, None, None
    for model_params in model_param_combos:
        trial_name = f"t2v_n{model_params['n_frequencies']}_mlp{model_params['mlp_dim']}_lr{model_params['lr']:.3e}"
        trial_best_loss, model_state = train_t2v(
            train_loader,
            val_loader,
            device,
            epochs=200,
            writer_dir=f"data/tensorboard/{run_start_ts}",
            trial_name=trial_name,
            **model_params
        )

        if trial_best_loss < best_loss:
            best_loss = trial_best_loss
            best_trial_name = trial_name
            best_params = model_params

        # Save the best model from this trial trial to disk
        th.save(model_state, f"data/t2v_weights/{trial_name}.pth")

    print(f"Best trial: {best_trial_name} with loss: {best_loss}")
    print(f"{"=" * 30}Parameters{"=" * 30}")
    dumps(best_params, indent=4, sort_keys=True)

    return best_trial_name


def try_load_t2v(model_weight_path: str = "t2v_n64_mlp1024_lr6.310e-05"):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    embedding = STEmbedding(
        6,
        64,
        pretrained_t2v=f"data/t2v_weights/{model_weight_path}.pth",
        time_idx=[0, 1, 2],
        ignore_cols=[3]
    ).to(device)

    # # Push a batch of N x T x F_t through the model
    # model.requires_grad_(False)
    # model.eval()
    dates = th.Tensor([[
        [2024, 11, 1],
        [2024, 11, 2],
        [2024, 11, 3],
        [2024, 11, 4],
        [2024, 11, 5],
        [2024, 11, 6],
        [2024, 11, 7],
        [2024, 11, 8],
        [2024, 11, 9],
        [2024, 11, 10],
    ]

    ])

    dates /= th.Tensor([3000, 12, 31])

    # Create a 2 x 10 x 3 tensor with random values to concatenate with the dates
    rand_data = th.rand(2, 10, 3)
    # expand the dates to match the shape of the random data
    dates = dates.expand(2, 10, 3)
    data = th.cat([dates, rand_data], dim=-1)

    outs = embedding(data.to(device))
    print(outs.shape)


if __name__ == '__main__':
    best_trial = run_pretraining()
    try_load_t2v()
