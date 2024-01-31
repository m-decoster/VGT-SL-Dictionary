import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .pose import PoseDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, num_workers: int, batch_size: int, data_dir: str, retrain_on_all: bool, **kwargs):
        """Initialize the data module.

        :param num_workers: The number of workers for the data loaders.
        :param batch_size: Batch size.
        :param data_dir: The root data directory.
        :param retrain_on_all: If provided, will load the validation set as part of the training set."""
        super(DataModule, self).__init__()

        # Arguments.
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Create datasets.
        self.train_set = PoseDataset('train', data_dir, retrain_on_all, kwargs['cut_transients'])
        self.val_set = PoseDataset('val', data_dir, retrain_on_all, kwargs['cut_transients'])
        self.test_set = PoseDataset('test', data_dir, retrain_on_all, kwargs['cut_transients'])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          shuffle=True, collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          collate_fn=collate)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          collate_fn=collate)

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument('--num_workers', type=int, help='Number of DataLoader workers.', default=0)
        parser.add_argument('--data_dir', type=str, help='Root dataset directory.', required=True)
        parser.add_argument('--cut_transients', action='store_true',
                            help='Cut off transient movements from the beginning and the end of signs.')
        return parent_parser


def collate(batch):
    clips = [e[0] for e in batch]
    targets = [e[1] for e in batch]
    filenames = [e[2] for e in batch]

    clips = torch.nn.utils.rnn.pad_sequence(clips, batch_first=False, padding_value=float("nan"))
    targets = torch.from_numpy(np.array(targets))

    return clips, targets, filenames
