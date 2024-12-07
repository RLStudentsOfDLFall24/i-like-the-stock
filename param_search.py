import datetime
import os
from itertools import product

from src.models import STTransformer
from src.training import run_grid_search


def run_st_grid_search():
    root = os.getcwd()
    t2v_weights = f"t2v_n64_mlp1024_lr6.310e-05"
    for symbol in [
        "atnf",
        "bivi",
        "cycc",
        "vtak"
    ]:
        print(f"{datetime.datetime.now()} | Running grid search for {symbol}")
        ctx_size = [20, 30]
        d_models = [64]
        batch_sizes = [64, 128]
        l_rates = [1e-5, 5e-5, 1e-4]
        fc_dims = [1024]
        fc_dropouts = [0.1]
        mlp_dims = [2048]
        mlp_dropouts = [0.3, 0.4]
        n_freqs = [64]
        num_encoders = [2, 3]
        num_heads = [4, 8]
        num_lstm_layers = [2]
        lstm_dim = [256]

        # use itertools.product to generate dictionaries of hyperparameters
        search_configs = [
            {
                "train_symbols": [
                    symbol,
                    # "atnf",
                    # "bivi",
                    # "cycc",
                    # "vtak",
                    "spx" # We only use SPX as part of the pretraining
                ],
                "target_symbol": symbol,
                "split": 0.30,
                "seq_len": ctx,
                "batch_size": bs,
                "trainer_params": {
                    "fine_tune_lr_ratio": 0.1,
                    "criterion": {
                        "name": "cb_focal",
                        "config": {
                            "gamma": 1.0,
                            "beta": 0.99,
                        },
                    },
                    "scheduler": {
                        "name": "plateau",
                    },
                    "optimizer": {
                        "name": "adam",
                        "config":{
                            "eps": 1e-8,
                            "weight_decay": 1e-3,
                            "betas": [0.9, 0.99]
                        }
                    },
                    "lr": lr,
                    "epochs": 200,
                },
                "model_params": {
                    "symbol": symbol,
                    "seq_len": ctx,
                    "d_model": d,
                    "time_idx": [6, 7, 8],
                    "ignore_cols": [0],
                    "fc_dim": fc,
                    "fc_dropout": fcd,
                    "mlp_dim": mlp,
                    "mlp_dropout": mld,
                    "k": k,
                    "num_encoders": ne,
                    "num_heads": nh,
                    "num_lstm_layers": nl,
                    "lstm_dim": ld,
                    "pretrained_t2v": f"{root}/data/t2v_weights/{t2v_weights}.pth",
                }
            }
            for d, lr, fc, fcd, mlp, mld, k, ne, nh, nl, ld, ctx, bs in product(
                d_models,
                l_rates,
                fc_dims,
                fc_dropouts,
                mlp_dims,
                mlp_dropouts,
                n_freqs,
                num_encoders,
                num_heads,
                num_lstm_layers,
                lstm_dim,
                ctx_size,
                batch_sizes,
            )
        ]

        run_grid_search(
            STTransformer,
            search_configs,
            trial_prefix=f"stt_{symbol}_candidate_search_wspx",
            root=root,
            y_lims=(0.45, 2.0),
        )
        print(f"{datetime.datetime.now()} | Finished grid search for {symbol}")


if __name__ == '__main__':
    run_st_grid_search()
