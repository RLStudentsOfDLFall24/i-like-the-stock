from collections import namedtuple
import logging
import sys
import os

import pandas as pd
import torch as th
from src.models.baselines import RiddlerModel
from src.models.rnn import RNN
from src.models.sttransformer import STTransformer
from src.models.lnn import LNN
from src.models.lnn_cfc import CfC_LNN
from src.training import run_experiment, get_spx_benchmark
from training_tools.utils import plot_simulation_result
from training_tools.eval import evaluate

import yaml
import numpy as np

MODEL_TYPES = {
    'rnn':RNN, 
    'transformer':STTransformer, 
    'lnn':LNN, 
    'lnn_cfc': CfC_LNN, 
    'baseline_tuned': RiddlerModel,
    'baseline_uniform': RiddlerModel,
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
                

    for key in configs:
        config_data = configs[key]
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
        perf_results = []
        for m in models:
            seq_len = m.trainer_params['seq_len']
            batch_size = m.trainer_params['batch_size']
            split = m.trainer_params['global_to_target_split']
            print('Starting training for ', m)
            eval_res = run_experiment(
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

            print('Avg Test Loss:',eval_res[4],
                '\nTest Accuracy:', eval_res[5],
                '\nF1:',eval_res[6],
                '\nPred Dist:',eval_res[7],
                '\nMCC:',eval_res[8],
                '\nAverage Epoch time:', eval_res[-1][0][:, 0].mean() if eval_res[-1][0].shape[0] > 0 else 0,
                '\nAverage Train time:', eval_res[-1][0][:, 1].mean() if eval_res[-1][0].shape[0] > 0 else 0,
                '\nAverage Validate time:', eval_res[-1][0][:, 2].mean() if eval_res[-1][0].shape[0] > 0 else 0,
                '\nTest time:', eval_res[-1][1],
                )

            sim_results.append(eval_res[-2])
            
            perf_results.append(np.concat((eval_res[-1][0].mean(0) if eval_res[-1][0].shape[0] > 0 else np.zeros(3), [eval_res[-1][1]])))

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
            fig_name=f"all_models_{target_symbol}_{key}",
        )


if __name__ == '__main__':
    # allows us to run a single config file (good for testing a one off or one experiment set)
    config = sys.argv[-1] if len(sys.argv) == 3 and '-c' == sys.argv[1] else None
    run(config)
