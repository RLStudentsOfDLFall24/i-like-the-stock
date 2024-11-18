from itertools import product

import numpy as np
import pandas as pd
import torch as th
from scipy.linalg.tests.test_fblas import accuracy
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook, tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from models.sttransformer import STTransformer
from src import create_datasets


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
    pb = tqdm(dataloader)

    losses = np.zeros(len(dataloader))
    # accuracies = np.zeros(len(dataloader))

    for ix, data in enumerate(pb):
        x = data[0].to(device)
        y = data[1].to(device)

        optimizer.zero_grad()
        logits = model(x)
        # If we only have one output, we need to unsqueeze the y tensor
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)

        loss = criterion(logits, y)

        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
        device: th.device = 'cpu',
) -> tuple[float, float, float, float]:
    # Set the model to evaluation
    model.eval()
    # total_loss = 0.0

    losses = np.zeros(len(dataloader))
    accuracies = np.zeros(len(dataloader))

    all_preds = []
    all_labels = []
    with th.no_grad():
        pb = tqdm(dataloader, ascii=True)
        for ix, data in enumerate(pb):
            x = data[0].to(device)
            y = data[1].to(device)

            logits = model(x)
            # Get the argmax of the logits
            preds = th.argmax(logits, dim=1)
            all_preds.append(preds)
            all_labels.append(y)

            losses[ix] = criterion(logits, y).item()
            accuracies[ix] = th.sum(th.argmax(logits, dim=1) == y).item() / y.shape[0]
            pb.set_description_str("Batch: %d, Loss: %.4f " % ((ix + 1), losses[ix].item()))

    # concat the preds and labels, then send to cpu
    all_preds = th.cat(all_preds).cpu()
    all_labels = th.cat(all_labels).cpu()

    # report = classification_report(all_labels, all_preds, output_dict=True)
    # print(report)
    # compute f1 score
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    # Use sklearn for pre
    # Print average accuracy


    # print(f"{label} Accuracy: {accuracies.mean():.4f}")
    return losses.sum(), losses.mean(), accuracies.mean(), f1_weighted


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
        epochs: int = 20
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
        time_idx=[0],
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
            optimizer = th.optim.Adam(model.parameters(), lr=lr)
        case "sgd":
            optimizer = th.optim.SGD(model.parameters(), lr=lr)
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    # Set the scheduler
    match scheduler:
        case "plateau":
            scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        case "step":
            scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        case "multi":
            scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15])
        case _:
            raise ValueError(f"Unknown scheduler: {scheduler}")

    match criterion:
        case "ce":
            criterion = th.nn.CrossEntropyLoss()
        case _:
            raise ValueError(f"Unknown criterion: {criterion}")

    train_losses = np.zeros(epochs)
    valid_losses = np.zeros(epochs)

    # change to tqdm progress bar
    # for epoch in range(epochs):
    for epoch in tqdm(range(epochs)):
        # Run training over the batches
        train_loss, train_loss_avg = train(model, train_loader, optimizer, criterion, device)
        scheduler.step(train_loss)

        # Evaluate the validation set
        valid_loss, valid_loss_avg, _, _ = evaluate(model, valid_loader, criterion, device)

        # Log the progress
        train_losses[epoch] = train_loss_avg
        valid_losses[epoch] = valid_loss_avg

    # Evaluate the test set
    test_loss, test_loss_avg, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)

    return train_losses, valid_losses, test_loss, test_loss_avg, test_acc, test_f1


def run_experiment(symbol: str, seq_len: int, batch_size: int, **kwargs):
    train_data, valid_data, test_data = create_datasets(symbol, seq_len=seq_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_sttransformer(
        x_dim=train_data[0][0].shape[1],
        seq_len=seq_len,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        **kwargs
    )


def run():
    th.manual_seed(1984)
    # We want to run a simple search over the hyperparameters
    ctx_size = [32]
    d_models = [64]
    l_rates = [5e-6]
    fc_dims = [2048]
    fc_dropouts = [0.3]
    n_freqs = [64]
    num_encoders = [2]
    num_heads = [4]
    num_lstm_layers = [2]
    lstm_dim = [256]

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
            "symbol": "cycc",
            "seq_len": ctx,
            "batch_size": 32,
            "d_model": d,
            "lr": lr,
            "fc_dim": fc,
            "fc_dropout": fcd,
            "k": k,
            "num_encoders": ne,
            "num_heads": nh,
            "num_lstm_layers": nl,
            "lstm_dim": ld,
            "optimizer": "adam",
            # "scheduler": "plateau",
            "scheduler": "multi",
            "criterion": "ce",
            "epochs": 50
        }
        for d, lr, fc, fcd, k, ne, nh, nl, ld, ctx in product(
            d_models,
            l_rates,
            fc_dims,
            fc_dropouts,
            n_freqs,
            num_encoders,
            num_heads,
            num_lstm_layers,
            lstm_dim,
            ctx_size
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
        "test_f1": []
    }

    for trial, config in enumerate(configurations):
        train_losses, v_losses, test_loss, test_loss_avg, test_acc, f1 = run_experiment(**config)
        # abstract this up level
        # plot train and valid losses on the same graph
        plt.plot(train_losses, label='Train Loss')
        plt.plot(v_losses, label='Valid Loss')
        # Set y scale between 0.25 and 1.25
        # plt.ylim(0.4, 3)
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

        # Iterate over the config and append the values to the dictionary
        for key, value in config.items():
            results_dict[key].append(value)

    # Save the results to a CSV file
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv("../data/st_trial_results.csv", index=False)


if __name__ == '__main__':
    run()
