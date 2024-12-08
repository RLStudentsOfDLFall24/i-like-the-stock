from collections import namedtuple
import logging
import sys
import os

import pandas as pd
import torch as th
from src.models.rnn import RNN
from src.models.sttransformer import STTransformer
from src.models.lnn import LNN
from src.models.lnn_cfc import CfC_LNN
from src.training import run_experiment, get_spx_benchmark, get_param_count
from training_tools.utils import plot_simulation_result

import yaml
import numpy as np

MODEL_TYPES = {
    'rnn':RNN, 
    'transformer':STTransformer, 
    'lnn':LNN, 
    'lnn_cfc': CfC_LNN, 
}
Model = namedtuple('Model', ['key', 'classname', 'params', 'trainer_params', 'device'])


def run(config_file: str = None):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    config_folder = './configs/'
    config_model_folder = './configs/models/'

    configs = {}
    if config_file is not None:
        with open(config_file, 'r') as f:
            configs[os.path.basename(config_file)] = yaml.safe_load(f)
    else:
        for entry in os.listdir(config_folder):
            full_path = os.path.join(config_folder, entry)
            if os.path.isfile(full_path) and (full_path.endswith('yml') or full_path.endswith('yaml')):
                with open(full_path, 'r') as f:
                    configs[os.path.basename(full_path)] = yaml.safe_load(f)
                
    experiment_data = {}
    for config_key in configs:
        config_data = configs[config_key]
        models = []

        optimizer_override = config_data['overrides']['optimizer'] if 'overrides' in config_data and 'optimizer' in config_data['overrides'] else None
        criterion_override = config_data['overrides']['criterion'] if 'overrides' in config_data and 'criterion' in config_data['overrides'] else None
        scheduler_override = config_data['overrides']['scheduler'] if 'overrides' in config_data and 'scheduler' in config_data['overrides'] else None
        global_to_target_split = config_data['overrides']['global_to_target_split'] if 'overrides' in config_data and 'global_to_target_split' in config_data['overrides'] else None

        for m in config_data['run_models']:
            if m not in MODEL_TYPES:
                logging.error('Model type must be one of: ' + str(MODEL_TYPES))
                continue
            
            with open(f'{config_model_folder}{m}.yml') as f:
                model_params = yaml.safe_load(f)

            params = model_params['params']
            trainer_params = model_params['trainer']

            if global_to_target_split is not None:
                trainer_params['global_to_target_split'] = global_to_target_split
            if optimizer_override is not None:
                trainer_params['optimizer'] = optimizer_override
            if criterion_override is not None:
                trainer_params['criterion'] = criterion_override
            if scheduler_override is not None:
                trainer_params['scheduler'] = scheduler_override
            
            models.append(Model(m, MODEL_TYPES[m], params, trainer_params, device))

        log_splits = config_data['training_params']['log_splits']
        train_symbols = config_data['training_params']['train_symbols']
        target_symbol = config_data['training_params']['target_symbol']

        sim_results = []
        perf_results = {
            'Experiment': [],
            'Model': [],
            'Param Count': [],
            'Pretrain Avg Test Loss': [],
            'Avg Test Loss': [],
            'Test Accuracy': [],
            'F1': [],
            'Pred Dist': [],
            'MCC': [],
            'Pretrain Total Training time': [],
            'Pretrain Average Epoch time': [],
            'Pretrain Average Train time': [],
            'Pretrain Average Validate time': [],
            'Pretrain Test time': [],
            'Total Training time': [],
            'Average Epoch time': [],
            'Average Train time': [],
            'Average Validate time': [],
            'Test time': [],
        }
        for m in models:
            seq_len = m.trainer_params['seq_len']
            batch_size = m.trainer_params['batch_size']
            split = m.trainer_params['global_to_target_split']
            print('Starting training for ', m)
            (model, _, _, _, avg_test_loss, test_acc, f1, pred_dist, mcc, simulate, times, test_times), \
                (pre_avg_test_loss, pre_times, pre_test_times) = run_experiment(
                model=m.classname,
                train_symbols=train_symbols,
                target_symbol=target_symbol,
                seq_len=seq_len,
                batch_size=batch_size,
                log_splits=log_splits,
                model_params=m.params,
                trainer_params=m.trainer_params,
                seed=1984,
                split=split,
                key=m.key)

            perf_results['Model'].append(m.key)
            perf_results['Param Count'].append(get_param_count(model))
            perf_results['Avg Test Loss'].append(avg_test_loss)
            perf_results['Pretrain Avg Test Loss'].append(pre_avg_test_loss)
            perf_results['Test Accuracy'].append(test_acc)
            perf_results['F1'].append(f1)
            perf_results['Pred Dist'].append(pred_dist.numpy())
            perf_results['MCC'].append(mcc)
            perf_results['Total Training time'].append(times[:, 1].sum() if times.shape[0] > 0 else 0)
            perf_results['Average Epoch time'].append(times[:, 0].mean() if times.shape[0] > 0 else 0)
            perf_results['Average Train time'].append(times[:, 1].mean() if times.shape[0] > 0 else 0)
            perf_results['Average Validate time'].append(times[:, 2].sum() if times.shape[0] > 0 else 0)
            perf_results['Test time'].append(test_times)
            perf_results['Pretrain Total Training time'].append(pre_times[:, 1].sum() if pre_times.shape[0] > 0 else 0)
            perf_results['Pretrain Average Epoch time'].append(pre_times[:, 0].mean() if pre_times.shape[0] > 0 else 0)
            perf_results['Pretrain Average Train time'].append(pre_times[:, 1].mean() if pre_times.shape[0] > 0 else 0)
            perf_results['Pretrain Average Validate time'].append(pre_times[:, 2].sum() if pre_times.shape[0] > 0 else 0)
            perf_results['Pretrain Test time'].append(pre_test_times)

            print(
                  'Avg Test Loss:', perf_results['Avg Test Loss'][-1],
                '\nTest Accuracy:', perf_results['Test Accuracy'][-1],
                '\nF1:',perf_results['F1'][-1],
                '\nPred Dist:', perf_results['Pred Dist'][-1],
                '\nMCC:', mcc,
                '\nTotal Train time:', times[:, 0].sum() if times.shape[0] > 0 else 0,
                '\nAverage Epoch time:', times[:, 0].mean() if times.shape[0] > 0 else 0,
                '\nAverage Train time:', times[:, 1].mean() if times.shape[0] > 0 else 0,
                '\nAverage Validate time:', times[:, 2].mean() if times[0].shape[0] > 0 else 0,
                '\nTest time:', test_times,
                )

            sim_results.append(simulate)
        
        for key in perf_results:
            perf_results[key] = np.array(perf_results[key])
        experiment_data[config_key] = perf_results
        
        # Merge simulations, keep only one of the symbol price columns
        sim_df = pd.concat(sim_results, axis=1)
        not_dupes = ~sim_df.columns.duplicated()
        sim_df = sim_df.loc[:, not_dupes]

        # Add the spx benchmark
        spx_bench = get_spx_benchmark(root='.')
        sim_df = pd.concat([sim_df, spx_bench], axis=1)

        plot_simulation_result(
            sim_df,
            fig_title=f"Strategy Results | {target_symbol}",
            fig_name=f"all_models_{target_symbol}_{config_key}",
        )

    pd_data = None
    for key in experiment_data:
        experiment = experiment_data[key]
        experiment['Experiment'] = [key] * experiment['Test time'].shape[0]
        pred_dist = experiment['Pred Dist']

        experiment['Pred Dist: Sell'] = pred_dist[:, 0]
        experiment['Pred Dist: Hold'] = pred_dist[:, 1]
        experiment['Pred Dist: Buy'] = pred_dist[:, 2]

        del experiment['Pred Dist']

        df = pd.DataFrame(experiment)
        pd_data = pd.concat(pd_data, df) if pd_data is not None else df

    pd_data.to_csv('./data/collected_experimental_data.csv', index=False)


if __name__ == '__main__':
    # allows us to run a single config file (good for testing a one off or one experiment set)
    config = sys.argv[-1] if len(sys.argv) == 3 and '-c' == sys.argv[1] else None
    run(config)
