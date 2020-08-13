"""
Example template for defining a system.
Taken from: https://github.com/PyTorchLightning/pytorch-lightning/blob/0.8.5/pl_examples/models/lightning_template.py
With minor modifications to decouple nn.Module from LightningModule
"""
import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning.core import LightningModule
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class ImageClassificationExperiment(LightningModule):
    """
    Sample model to show how to define a template.

    Example:

        >>> # define a simple Net
        >>> params = dict(
        ...     drop_prob=0.2,
        ...     train_batch_size=2,
        ...     eval_batch_size=4,
        ...     in_features=28 * 28,
        ...     learning_rate=0.001,
        ...     optimizer_name='adam',
        ...     data_root='./datasets',
        ...     out_features=10,
        ...     num_workers=4,
        ...     hidden_dim=1000,
        ... )
        >>> model = ImageClassificationExperiment(**params)
    """

    def __init__(self,
                 model,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 64,
                 learning_rate: float = 0.001,
                 optimizer_name: str = 'adam',
                 data_root: str = './datasets',
                 num_workers: int = 4,
                 hidden_dim: int = 1000,
                 **kwargs
                 ):
        # init superclass
        super().__init__()

        self.num_workers = num_workers

        # self.drop_prob = drop_prob
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        # self.in_features = in_features
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.data_root = data_root
        # self.out_features = out_features
        self.hidden_dim = hidden_dim

        self.model = model

        self.train_set, self.train_eval_set, self.val_set, self.test_set = None, None, None, None

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, data_loader_index):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    @staticmethod
    def aggregate_results(results):
        avg_loss = torch.stack([x['loss'] for x in results]).mean()
        acc = sum([x['n_correct_pred'] for x in results]) / sum(x['n_pred'] for x in results)
        return avg_loss, acc

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """

        train_eval_result = self.aggregate_results(outputs[0])
        valid_result = self.aggregate_results(outputs[1])
        tensorboard_logs = {
            'train_eval_loss': train_eval_result[0], 'train_eval_acc': train_eval_result[1],
            'valid_loss': valid_result[0], 'valid_acc': valid_result[1],
        }
        return {'val_loss': valid_result[0], 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        valid_result = self.aggregate_results(outputs[0])
        test_result = self.aggregate_results(outputs[1])
        tensorboard_logs = {
            'valid_loss': valid_result[0], 'valid_acc': valid_result[1],
            'test_loss': test_result[0], 'test_acc': test_result[1],
        }
        return {'test_loss': test_result[0], 'log': tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        CIFAR10(self.data_root, train=True, download=True, transform=transforms.ToTensor())
        CIFAR10(self.data_root, train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

        train_set = CIFAR10(self.data_root, train=True, download=False, transform=transform)
        train_len = int(len(train_set) * .8)
        val_len = len(train_set) - train_len
        self.train_set, self.val_set = torch.utils.data.random_split(train_set, [train_len, val_len])
        self.test_set = CIFAR10(self.data_root, train=False, download=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return [
            DataLoader(self.train_set, batch_size=self.eval_batch_size, num_workers=self.num_workers),
            DataLoader(self.val_set, batch_size=self.eval_batch_size, num_workers=self.num_workers)]

    def test_dataloader(self):
        return [
            DataLoader(self.val_set, batch_size=self.eval_batch_size, num_workers=self.num_workers),
            DataLoader(self.test_set, batch_size=self.eval_batch_size, num_workers=self.num_workers)]

    @staticmethod
    def add_experiment_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'cifar10'), type=str)
        parser.add_argument('--num_workers', default=8, type=int)

        # optimization hyperparameters
        parser.add_argument('--epochs', default=50, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--train_batch_size', default=1024 * 8, type=int)
        parser.add_argument('--eval_batch_size', default=1024 * 8, type=int)

        return parser
