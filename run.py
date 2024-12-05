from collections import namedtuple

import pandas as pd
import torch as th
from src.models.rnn import RNN
from src.models.sttransformer import STTransformer
from src.models.lnn import LNN
from src.training import run_experiment, get_spx_benchmark
from training_tools.utils import plot_simulation_result

import yaml


MODEL_TYPES = {'rnn':RNN, 'transformer':STTransformer, 'lnn':LNN}
Model = namedtuple('Model', ['key', 'classname', 'params', 'trainer_params', 'device'])


def run():
    # TODO - setup training loop after we read config
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    with open('config.yml', 'r') as f:
        config_data = yaml.safe_load(f)

    models = []

    for m in config_data['model_types']:
        if m not in MODEL_TYPES:
            raise ValueError('Model type must be one of: ' + str(MODEL_TYPES))
        params = config_data[m]
        trainer_params = config_data[m+'_trainer']
        models.append(Model(m, MODEL_TYPES[m], params, trainer_params, device))

    if 'train' in config_data['mode']:
        log_splits = config_data['global_params']['log_splits']
        train_symbols = config_data['global_params']['train_symbols']
        target_symbol = config_data['global_params']['target_symbol']

        sim_results = []
        for m in models:
            seq_len = trainer_params['seq_len']
            batch_size = trainer_params['batch_size']
            split = config_data['global_params']['global_to_target_split']
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
                split=split)

            print('Avg Test Loss:',eval_res[4],
                '\nTest Accuracy:', eval_res[5],
                  '\nF1:',eval_res[6],
                  '\nPred Dist:',eval_res[7])

            sim_results.append(eval_res[-1])
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
            fig_name=f"all_models_{target_symbol}",
        )

    if 'eval' in config_data['mode']:
        for m in models:
            print('Running eval for ', m)
            #fwd = m(th.Tensor(10, 10))

            #print(fwd)
            pass
    pass


if __name__ == '__main__':
    run()
