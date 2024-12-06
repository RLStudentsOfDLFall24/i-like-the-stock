from datetime import datetime

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import print_target_distribution
from src.dataset._dataset_utils import create_datasets
from src.models.abstract_model import AbstractModel
from training_tools import get_criterion, get_optimizer, get_scheduler, train, evaluate, plot_results, \
    plot_simulation_result
from training_tools.utils import get_data


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
        symbol: str = "default",
        **kwargs
) -> tuple[AbstractModel, np.ndarray, np.ndarray, float, float, float, float, th.Tensor, float, pd.DataFrame]:
    """Train a model and test the methods"""
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    epochs: int = trainer_params['epochs']
    opt_type = trainer_params['optimizer']['name']
    lr = trainer_params['lr']
    crit_type = trainer_params['criterion']['name']
    add_loss_to_scheduler = trainer_params['scheduler']['name'] == 'plateau'

    if model is None:
        model = model_class(
            d_features=x_dim,
            device=device,
            **model_params
        )

    # Set the optimizer
    optimizer = get_optimizer(
        opt_type,
        model=model,
        lr=lr,
        config=trainer_params['optimizer']['config']
    )

    # Set the scheduler
    scheduler = get_scheduler(
        trainer_params['scheduler']['name'],
        optimizer,
        config=trainer_params['scheduler']['config'] if 'config' in trainer_params[
            'scheduler'] else {}
    )

    # Set the criterion
    criterion = get_criterion(crit_type, train_label_ct, trainer_params, device)
    train_losses = np.zeros(epochs)
    valid_losses = np.zeros(epochs)

    for epoch in range(epochs):
        # Run training over the batches
        _, train_loss_avg = train(model, train_loader, optimizer, criterion, device, epoch, writer=writer)
        # Evaluate the validation set
        _, v_loss_avg, v_acc, v_f1, v_pred_dist, v_mcc, _ = evaluate(model, valid_loader, criterion, device=device)

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
        if add_loss_to_scheduler:
            scheduler.step(v_loss_avg)
        else:
            scheduler.step()

        # Update the progress bar to also show the loss
        pred_string = " - ".join([f"C{ix} {x:.3f}" for ix, x in enumerate(v_pred_dist)])
        print(f"E: {epoch + 1} | Train: {train_loss_avg:.4f} | Valid: {v_loss_avg:.4f} | V_Pred Dist: {pred_string}")

    # Evaluate the test set
    test_loss, test_loss_avg, test_acc, test_f1, test_pred_dist, test_mcc, sim_df = evaluate(
        model,
        test_loader,
        criterion,
        device=device,
        simulate=True
    )
    # rename the value column to be the model type
    model_name = model.__class__.__name__[0:3]
    sim_df.rename(
        columns={"value": model_name, "price": symbol},
        inplace=True
    )

    # Add first three characters of the model name to the DataFrame
    # sim_df["model"] = model.__class__.__name__[0:3]

    if writer is not None:
        writer.add_scalar("Loss/test", test_loss_avg, epochs)
        writer.add_scalar("Accuracy/test", test_acc, epochs)
        writer.add_scalar("F1/test", test_f1, epochs)
        writer.add_scalar("MCC/test", test_mcc, epochs)

        # We can also compute cumulative returns and add that to the tensorboard
        cum_ret = (sim_df[model_name].iloc[-1] - sim_df[model_name].iloc[0]) / sim_df[model_name].iloc[0]
        writer.add_scalar("Simulation/Cumulative Return", cum_ret, epochs)

    return model, train_losses, valid_losses, test_loss, test_loss_avg, test_acc, test_f1, test_pred_dist, test_mcc, sim_df


def run_experiment(
        model: type[AbstractModel],
        train_symbols: list[str],
        target_symbol: str,
        seq_len: int,
        batch_size: int,
        log_splits: bool = False,
        model_params: dict = None,
        trainer_params: dict = None,
        split: float = 1.0,
        root: str = ".",
        seed: int = 0,
        **kwargs
) -> tuple[AbstractModel, np.ndarray, np.ndarray, float, float, float, float, th.Tensor, float, pd.DataFrame]:
    """
    Load the data symbols and create PriceSeriesDatasets.

    The symbols data is loaded into PriceSeriesDatasets, augmented with indicators
    and then partitioned into train, validation, and test sets. The data are then
    wrapped in DataLoader objects for use in training and evaluation loops.

    :param model: The model class to train
    :param train_symbols: The symbols to use for training
    :param target_symbol: The target symbol to use for testing
    :param seq_len: The sequence length to use
    :param batch_size: The batch size to use for DataLoader
    :param log_splits: Whether to log the target distribution
    :param model_params: Additional arguments for the model
    :param trainer_params: Additional arguments for the trainer
    :param split: The split ratio for training and fine-tuning
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
        - sim_df: The simulation results DataFrame
    """

    if seed > 0:
        th.manual_seed(seed)

    target_train, train_label_ct, train_loader, valid_loader, test_loader = get_data(train_symbols, batch_size, seq_len, target_symbol, log_splits, root)

    epochs = trainer_params['epochs']
    trainer_params['epochs'] = int(split * epochs)

    pretrain = train_model(
        model=None,
        x_dim=target_train.feature_dim,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_label_ct=train_label_ct,
        model_class=model,
        model_params=model_params,
        trainer_params=trainer_params
    )

    trainer_params['epochs'] = int((1 - split) * epochs)
    train_loader = DataLoader(target_train, batch_size=batch_size, shuffle=True)
    train_label_ct = target_train.target_counts
    trainer_params['lr'] = trainer_params['lr'] * trainer_params['fine_tune_lr_ratio']

    # finetune
    return train_model(
        model=pretrain[0],
        x_dim=target_train.feature_dim,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_label_ct=train_label_ct,
        model_class=model,
        model_params=model_params,
        trainer_params=trainer_params,
        symbol=target_symbol,
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
        "test_acc", "test_f1", "test_pred_dist", "test_mcc", "test_cum_ret"
    })
    results_dict = {key: [] for key in result_keys}

    # Begin main run loop
    run_start_ts = int(datetime.now().timestamp())
    for trial, config in enumerate(search_configs):
        criterion = config['trainer_params']['criterion']['name']
        writer_dir = f"{root}/data/tensorboard/{run_start_ts}/{criterion}/{trial_prefix}_{trial:03}"
        writer = SummaryWriter(log_dir=writer_dir) if use_writer else None
        _, tr_loss, v_loss, tst_loss, tst_loss_avg, tst_acc, tst_f1, tst_pred_dist, tst_mcc, sim = (run_experiment
            (
            model=model_type,
            log_splits=trial == 0,
            writer=writer,
            root=root,
            seed=1984,
            **config
        ))

        tgt_symbol = config['target_symbol']
        print_target_distribution([("Test", tst_pred_dist)])
        plot_results(tr_loss, v_loss, config['trainer_params']['epochs'], y_lims=y_lims, root=root,
                     image_name=f'{trial_prefix}_{tgt_symbol}_{trial:03}_loss')
        # Plot the simulation results
        plot_simulation_result(
            sim,
            fig_title=f"Simulation Results for {tgt_symbol}|{trial:03}",
            fig_name=f"{trial_prefix}_{trial:03}",
            root=root
        )

        cum_ret = (sim["STT"].iloc[-1] - sim["STT"].iloc[0]) / sim["STT"].iloc[0]

        # Save the loss and training results to the dictionary
        results_dict["trial_id"].append(trial)
        results_dict["test_loss"].append(tst_loss)
        results_dict["test_loss_avg"].append(tst_loss_avg)
        results_dict["test_acc"].append(tst_acc)
        results_dict["test_f1"].append(tst_f1)
        results_dict["test_mcc"].append(tst_mcc)
        results_dict["test_pred_dist"].append([round(x.item(), 3) for x in tst_pred_dist])
        results_dict["test_cum_ret"].append(cum_ret)
        # Iterate over the config and append the values to the dictionary
        for key, value in config["trainer_params"].items():
            results_dict[key].append(value)

        for key, value in config["model_params"].items():
            results_dict[key].append(value)

        results_dict["batch_size"].append(config["batch_size"])

    # Save the results to a CSV file
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f"{root}/data/{trial_prefix}_results.csv", index=False)


def get_spx_benchmark(root: str) -> pd.DataFrame:
    """
    Load spx data and compute normalized value over the test period.
    :param root: The root directory to load the data
    :return: A DataFrame of the normalized SPX value over the test period
    """
    _, _, spx_test = create_datasets(
        "spx",
        seq_len=10,  # seq_len doesn't matter for this
        fixed_scaling=[(7, 3000.), (8, 12.), (9, 31.)],
        log_splits=False,
        root=f"{root}/data/clean"
    )

    spx_prices = spx_test.unscaled_prices.detach().numpy()
    time_idx = pd.to_datetime(spx_test.time_idx.detach().numpy(), unit='s')
    res_df = pd.DataFrame(
        {
            "SPX": np.round(spx_prices / spx_prices[0], 2),
        }, index=time_idx)
    return res_df
