import torch
from src.models.rnn import RNN
from src.models.sttransformer import STTransformer
from src.models.lnn import LNN

import yaml

MODEL_TYPES = ['rnn','transformer','lnn']

with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)


models = []

for m in config_data['model_types']:
    if m not in MODEL_TYPES:
        raise ValueError('Model type must be one of: ' + str(MODEL_TYPES))
    params = config_data[m]

    if m == 'rnn':
        models.append(RNN(params['batch_size']))
    if m == 'transformer':
        models.append(STTransformer(params['batch_size']))
    if m == 'lnn':
        models.append(LNN(params['batch_size']))


if 'train' in config_data['mode']:
    for m in models:
        print('Starting training for ', m)
        #call train fn
        pass

if 'eval' in config_data['mode']:
    for m in models:
        print('Running eval for ', m)
        fwd = m(torch.Tensor(10,10))

        print(fwd)
