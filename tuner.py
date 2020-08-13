"""
Taken from
https://github.com/optuna/optuna/blob/master/examples/pytorch_lightning_simple.py

Optuna example that optimizes multi-layer perceptrons using PyTorch Lightning.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch Lightning, and MNIST. We optimize the neural network architecture. As it is too time
consuming to use the whole MNIST dataset, we here use a small subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly. Pruning can be turned on and off with the `--pruning` argument.
    $ python pytorch_lightning_simple.py [--pruning]


(2) Execute through CLI. Pruning is enabled automatically.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize pytorch_lightning_simple.py objective --n-trials=100 --study-name \
      $STUDY_NAME --storage sqlite:///example.db
"""
import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything

from classification_experiment import ImageClassificationExperiment
from mlp import MLP

import argparse
import os
import pkg_resources
import functools
import shutil
import trainer

import pytorch_lightning as pl
from pytorch_lightning import Callback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import inspect

import optuna
from optuna.integration import PyTorchLightningPruningCallback

PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")

pl.seed_everything(234)


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def objective(trial, **kwargs):
    # # Categorical parameter
    # optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])
    # # Int parameter
    # num_layers = trial.suggest_int('num_layers', 1, 3)
    # # Uniform parameter
    dropout_prob = trial.suggest_uniform('dropout_prob', 0.0, 1.0)
    # # Loguniform parameter
    # learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    # # Discrete-uniform parameter
    # drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.01)
    print("dropout_prob: {}".format(dropout_prob))
    kwargs.update(dropout_prob=dropout_prob)

    # Filenames for each trial must be made unique in order to access each checkpoint.
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"), monitor="val_acc"
    # )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We don't use any logger here as it requires us to implement several abstract
    # methods. Instead we setup a simple callback, that saves metrics from each validation step.
    metrics_callback = MetricsCallback()

    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT a model and the LIGHTNING Experiment class
    # ------------------------
    model = MLP(**kwargs)
    experiment = ImageClassificationExperiment(model=model, **kwargs)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    kwargs.update({"logger": False,
                   # "checkpoint_callback": checkpoint_callback,
                   "callbacks": [metrics_callback],
                   "early_stop_callback": PyTorchLightningPruningCallback(trial, monitor="val_loss")})

    valid_kwargs = inspect.signature(pl.Trainer.__init__).parameters
    trainer_kwargs = dict((name, kwargs[name]) for name in valid_kwargs if name in kwargs)

    trainer = pl.Trainer(**trainer_kwargs)
    # ------------------------
    # 3 START TRAINING
    # ------------------------

    trainer.fit(experiment)

    return metrics_callback.metrics[-1]["val_loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")

    parser.add_argument("--n_trials", type=int, default=100)

    parser = trainer.add_args(parser)
    args = parser.parse_args()
    kwargs = vars(args)

    parser.set_defaults(precision=16)

    pruner = optuna.pruners.HyperbandPruner(min_resource=5, max_resource=50, reduction_factor=3)

    objective_with_kwargs = functools.partial(objective, **kwargs)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective_with_kwargs, n_trials=kwargs["n_trials"])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(MODEL_DIR)
