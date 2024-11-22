import torch as th
from src.models.rnn import RNN
from src.models.sttransformer import STTransformer
from src.models.lnn import LNN

import yaml

MODEL_TYPES = ['rnn', 'transformer', 'lnn']


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
        # TODO push the model instantiation down to the experiment, we can pass a class and params
        d_features = 42

        if m == 'rnn':
            models.append(RNN(params['batch_size']))
        if m == 'transformer':
            models.append(STTransformer(**params, device=device))
        if m == 'lnn':
            models.append(LNN(params['batch_size']))

    if 'train' in config_data['mode']:
        for m in models:
            print('Starting training for ', m)
            # call train fn
            pass

    if 'eval' in config_data['mode']:
        for m in models:
            print('Running eval for ', m)
            fwd = m(th.Tensor(10, 10))

            print(fwd)
    pass


if __name__ == '__main__':
    run()
