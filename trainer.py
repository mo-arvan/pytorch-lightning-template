"""
Runs a model on a single node on GPU(s).
"""
import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything

from classification_experiment import ImageClassificationExperiment
from mlp import MLP

seed_everything(234)


def train(args):
    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT a model and the LIGHTNING Experiment class
    # ------------------------
    model = MLP(**vars(args))
    experiment = ImageClassificationExperiment(model=model, **vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(experiment)


def add_args(parent_parser=None):
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    if parent_parser is None:
        parent_parser = ArgumentParser(add_help=True)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = parent_parser

    # each LightningModule defines arguments relevant to it
    parser = MLP.add_model_specific_args(parser, root_dir)
    parser = ImageClassificationExperiment.add_experiment_specific_args(parser, root_dir)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1)
    return parser


def main():
    # Adding model, training and optimization arguments
    parser = add_args()
    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    train(args)


if __name__ == '__main__':
    main()
