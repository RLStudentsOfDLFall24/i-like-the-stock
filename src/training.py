import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

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
) -> tuple[float, float, float, float, th.Tensor, float]:
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
    mcc = matthews_corrcoef(all_labels, all_preds)

    return losses.sum(), losses.mean(), accuracies.mean(), f1_weighted, pred_dist, mcc


def train_model(
        model: AbstractModel,
        x_dim: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        train_label_ct: th.Tensor = None,
        model_class: type[AbstractModel] = None,
        writer: SummaryWriter = None,
        model_params: dict = None,
        trainer_params: dict = None,
        **kwargs
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, th.Tensor, float]:
    """Train a model and test the methods"""
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    epochs = trainer_params['epochs']
    opt_type = trainer_params['optimizer']
    lr = trainer_params['lr']
    crit_type = trainer_params['criterion']

    if model is None:
        model = model_class(
            d_features=x_dim,
            device=device,
            **model_params
        )

    # Set the optimizer
    match opt_type:
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
            raise ValueError(f"Unknown optimizer: {opt_type}")

    # Set the scheduler
    match trainer_params['scheduler']:
        case "plateau":
            scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5)
        case "step":
            scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        case "multi":
            scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15])
        case _:
            raise ValueError(f"Unknown scheduler: {trainer_params['scheduler']}")

    match crit_type:
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
                gamma=trainer_params['cbf_gamma'],
                beta=trainer_params['cbf_beta']
            )
        case _:
            raise ValueError(f"Unknown criterion: {crit_type}")

    epochs = trainer_params['epochs']

    train_losses = np.zeros(epochs)
    valid_losses = np.zeros(epochs)

    for epoch in range(epochs):
        # Run training over the batches
        train_loss, train_loss_avg = train(model, train_loader, optimizer, criterion, device, epoch, writer=writer)
        # Evaluate the validation set
        v_loss, v_loss_avg, v_acc, v_f1, v_pred_dist, v_mcc = evaluate(model, valid_loader, criterion, device=device)

        # Log the progress
        train_losses[epoch] = train_loss_avg
        valid_losses[epoch] = v_loss_avg

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss_avg, epoch)
            writer.add_scalar("Loss/valid", v_loss_avg, epoch)
            writer.add_scalar("Accuracy/valid", v_acc, epoch)
            writer.add_scalar("F1/valid", v_f1, epoch)
            writer.add_scalar("MCC/valid", v_mcc, epoch)

        # Update the learning rate scheduler on the average validation loss
        scheduler.step(v_loss_avg)
        # Update the progress bar to also show the loss
        pred_string = " - ".join([f"C{ix} {x:.3f}" for ix, x in enumerate(v_pred_dist)])
        print(
            f"E: {epoch + 1} | Train: {train_loss_avg:.4f} | Valid: {v_loss_avg:.4f} | V_Pred Dist: {pred_string}")

    # Evaluate the test set
    test_loss, test_loss_avg, test_acc, test_f1, test_pred_dist, test_mcc = evaluate(
        model,
        test_loader,
        criterion,
        device=device
    )
    if writer is not None:
        writer.add_scalar("Loss/test", test_loss_avg, epochs)
        writer.add_scalar("Accuracy/test", test_acc, epochs)
        writer.add_scalar("F1/test", test_f1, epochs)
        writer.add_scalar("MCC/test", test_mcc, epochs)

    return model, train_losses, valid_losses, test_loss, test_loss_avg, test_acc, test_f1, test_pred_dist


def run_experiment(model: type[AbstractModel],
                   train_symbols: list[str],
                   target_symbol: str,
                   seq_len: int,
                   batch_size: int,
                   log_splits: bool = False,
                   model_params: dict = None,
                   trainer_params: dict=None,
                   split: float = 0,
                   root: str = ".",
                   **kwargs) -> tuple[AbstractModel, np.ndarray, np.ndarray, float, float, float, float, th.Tensor, float]:
    """
    Load the data symbols and create PriceSeriesDatasets.

    The symbols data is loaded into PriceSeriesDatasets, augmented with indicators
    and then partitioned into train, validation, and test sets. The data are then
    wrapped in DataLoader objects for use in training and evaluation loops.

    :param seq_len: The sequence length to use
    :param batch_size: The batch size to use for DataLoader
    :param log_splits: Whether to log the target distribution
    :param model_params: Additional arguments for the model
    :param trainer_params: Additional arguments for the trainer
    :param root: The root directory to save the results
    :param kwargs: Additional arguments
    :return: A tuple of results as follows:
        - train_losses: The training losses for each epoch
        - valid_losses: The validation losses for each epoch
        - test_loss: The test loss
        - test_loss_avg: The average test loss
        - test_acc: The test accuracy
        - test_f1: The test F1 score
        - test_pred_dist: The test prediction distribution
        - mcc: The test Matthews correlation coefficient
    """

    trains = []
    target_train = None
    target_valid = None
    target_test = None
    th.manual_seed(1984)

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

    train_loader = DataLoader(concatted_trains, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(target_valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(target_test, batch_size=batch_size, shuffle=False)


    train_label_ct = th.sum(th.stack([x.target_counts for x in trains]),dim=0)
    #train_label_ct = concatted_trains.target_counts

    epochs = trainer_params['epochs']
    trainer_params['epochs'] = int(split*epochs)


    pretrain = train_model(
        model=None,
        x_dim=train_data[0][0].shape[1],
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_label_ct=train_label_ct,
        model_class=model,
        model_params=model_params, 
        trainer_params=trainer_params
    )

    trainer_params['epochs'] = int((1-split)*epochs)
    train_loader = DataLoader(target_train, batch_size=batch_size, shuffle=True)
    train_label_ct = target_train.target_counts
    trainer_params['lr'] = trainer_params['lr'] * trainer_params['fine_tune_lr_ratio']

    #finetune
    return train_model(
        model=pretrain[0],
        x_dim=train_data[0][0].shape[1],
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_label_ct=train_label_ct,
        model_class=model,
        model_params=model_params,
        trainer_params=trainer_params,
        **kwargs
    )


def run_grid_search(
        model_type: type[AbstractModel],
        search_configs: list[dict],
        trial_prefix: str = "default_trial",
        use_writer: bool = True,
        root: str = ".",
        y_lims: tuple[float, float] = (0.0, 2.0),
):
    """
    Run a grid search over configurations and save CSV results file.

    This method runs a simple grid search over the configurations and saves the
    results to a CSV file for future analysis. One may optionally use a tensorboard
    writer to log the results of the training process and view them in a tensorboard.
    A simple training/validation loss curve is plotted for each configuration and
    saved to the figures directory. The trial_id is saved as part of the CSV results
    and can allow for identification of the related plots.

    :param model_type: The model class to train
    :param search_configs: A list of dictionaries containing the configurations
    :param trial_prefix: The prefix to use for the trial
    :param use_writer: Whether to use a SummaryWriter
    :param root: The root directory to save the results
    :param y_lims: The y limits for the loss plot
    """
    # Get all the unique key names from trainer and model params in a list or set
    result_keys = {key for config in search_configs for key in config["trainer_params"].keys()}
    result_keys.update({key for config in search_configs for key in config["model_params"].keys()})
    # Keys we know we'll need
    result_keys.update({
        "batch_size", "symbol", "trial_id", "test_loss", "test_loss_avg",
        "test_acc", "test_f1", "test_pred_dist", "test_mcc"
    })
    results_dict = {key: [] for key in result_keys}

    # Begin main run loop
    run_start_ts = int(datetime.now().timestamp())
    for trial, config in enumerate(search_configs):
        criterion = config['trainer_params']['criterion']
        writer_dir = f"{root}/data/tensorboard/{run_start_ts}/{criterion}/{trial_prefix}_{trial:03}"
        writer = SummaryWriter(log_dir=writer_dir) if use_writer else None
        tr_loss, v_loss, tst_loss, tst_loss_avg, tst_acc, tst_f1, tst_pred_dist, tst_mcc = (run_experiment
            (
            model=model_type,
            log_splits=trial == 0,
            writer=writer,
            root=root,
            **config
        ))

        print_target_distribution([("Test", tst_pred_dist)])

        # Create the figures directory if it doesn't exist
        if not os.path.exists(f"{root}/figures"):
            print("Creating figures directory...")
            os.makedirs(f"{root}/figures")

        plt.plot(tr_loss, label='Train Loss')
        plt.plot(v_loss, label='Valid Loss')
        plt.xlim(0, config['trainer_params']['epochs'])
        plt.ylim(*y_lims)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{root}/figures/{trial_prefix}_{config['symbol']}_{trial:03}_loss.png")
        plt.close()

        # Save the loss and training results to the dictionary
        results_dict["trial_id"].append(trial)
        results_dict["test_loss"].append(tst_loss)
        results_dict["test_loss_avg"].append(tst_loss_avg)
        results_dict["test_acc"].append(tst_acc)
        results_dict["test_f1"].append(tst_f1)
        results_dict["test_mcc"].append(tst_mcc)
        results_dict["test_pred_dist"].append([round(x.item(), 3) for x in tst_pred_dist])


        # Iterate over the config and append the values to the dictionary
        for key, value in config["trainer_params"].items():
            results_dict[key].append(value)

        for key, value in config["model_params"].items():
            results_dict[key].append(value)

        results_dict["batch_size"].append(config["batch_size"])

    # Save the results to a CSV file
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f"{root}/data/{trial_prefix}_results.csv", index=False)
