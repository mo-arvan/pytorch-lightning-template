from argparse import ArgumentParser

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, drop_prob: float = 0.2, in_features: int = 28 * 28,
                 out_features: int = 10, hidden_dim: int = 1000,
                 **kwargs
                 ):
        super().__init__()

        self.drop_prob = drop_prob
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim

        self.c_d1 = nn.Linear(in_features=self.in_features,
                              out_features=self.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(self.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.drop_prob)

        self.c_d2 = nn.Linear(in_features=self.hidden_dim,
                              out_features=self.out_features)

    def forward(self, x):
        x = self.c_d1(x.view(x.size(0), -1))
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)
        x = self.c_d2(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # network params
        parser.add_argument('--in_features', default=32 * 32 * 3, type=int)
        parser.add_argument('--out_features', default=10, type=int)
        parser.add_argument('--hidden_dim', default=500, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        return parser
