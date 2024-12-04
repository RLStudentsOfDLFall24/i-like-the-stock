from collections import namedtuple

import pandas as pd
import torch as th
from src.models.rnn import RNN
from src.models.sttransformer import STTransformer
from src.models.lnn import LNN
from src.training import run_experiment
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
        symbol = config_data['global_params']['symbol']

        sim_results = []
        for m in models:
            seq_len = config_data[m.key]['seq_len']
            batch_size = config_data[m.key]['batch_size']
            print('Starting training for ', m)
            eval_res = run_experiment(model=m.classname, symbol=symbol, seq_len=seq_len, batch_size=batch_size, log_splits=log_splits, model_params=m.params, trainer_params=m.trainer_params)
            # TODO Look at the eval_res and see if we can save all the sim_df frames to plot all test simulations together
            print('Avg Test Loss:',eval_res[3], '\nTest Accuracy:', eval_res[4], '\nF1:',eval_res[5], '\nPred Dist:',eval_res[6])
            sim_results.append(eval_res[8])
        # TODO - merge all sim_results and plot them using utility
        # sim_results will have 3 dataframes with 2 columns each. we want a new dataframe with the first column from each and the same index
        sim_df = pd.concat(sim_results, axis=1)
        not_dupes = ~sim_df.columns.duplicated()
        sim_df = sim_df.loc[:, not_dupes]

        # pull in the util
        plot_simulation_result(
            sim_df,
            fig_title=f"Simulation Results for {symbol}",
            fig_name=f"all_models_{symbol}",
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
