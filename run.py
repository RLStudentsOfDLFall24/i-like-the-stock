import torch
from src.models.rnn import RNN
from src.models.transformer import Transformer
from src.models.lnn import LNN

import argparse

MODEL_TYPES = ['rnn','transformer','lnn']

parser = argparse.ArgumentParser(prog="SupervisedFinancialModelEvaluator",
                                 description="Evaluate multiple supervised learning models on stock ticker data",
                                 )
parser.add_argument('-m', '--modeltype', help=str(MODEL_TYPES))
parser.add_argument('-a', '--all', action='store_true')

args = parser.parse_args()


if args.all or (not args.all and args.modeltype == None):
    print('Running all models')
    all_models = True
else:
    if args.modeltype not in MODEL_TYPES:
        raise ValueError('Must be one of: ' + str(MODEL_TYPES) + ', or all')


models = []

if args.modeltype == 'rnn' or all_models:
    models.append(RNN())
if args.modeltype == 'transformer' or all_models:
    models.append(Transformer())
if args.modeltype == 'lnn' or all_models:
    models.append(LNN())


for m in models:
    fwd = m(torch.Tensor(10,10))

    print(fwd)
