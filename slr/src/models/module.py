"""PyTorch Lightning module definition. Delegates computation to one of the defined networks."""
from typing import List

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .poseformer import PoseFormer


class Module(pl.LightningModule):
    """Pytorch Lightning module that delegates to neural networks.

    IT IS ASSUMED THAT ALL NETWORKS ARE BATCH_FIRST=FALSE!"""

    def __init__(self, batch_size: int, learning_rate: float, weight_decay: float,
                 num_attention_layers: int, num_attention_heads: int, d_hidden: int, num_classes: int,
                 retrain_on_all: bool, lr_steps: List[int], pf_dropout: float, d_pose: int, **kwargs):
        """Create a Module.

        :param batch_size: Batch size.
        :param learning_rate: Initial learning rate.
        :param weight_decay: Weight decay.
        :param num_attention_layers: Number of self-attention layers.
        :param num_attention_heads: Number of self-attention heads in every layer.
        :param d_hidden: Size of the embeddings.
        :param num_classes: List of the number of classes in every language dataset.
        :param retrain_on_all: Whether to train on all training and validation data.
        :param lr_steps: (If retrain_on_all is true) when to reduce the learning rate.
        :param pf_dropout: Dropout value.
        :param d_pose: The amount of input keypoint values."""
        super(Module, self).__init__()

        # Hyperparameters / arguments.
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay if weight_decay is not None else 0.0
        self.num_classes = num_classes
        self.d_hidden = d_hidden
        self.retrain_on_all = retrain_on_all
        self.lr_steps = lr_steps
        self.pf_dropout = pf_dropout
        self.d_pose = d_pose
        self.num_attention_heads = num_attention_heads
        self.num_attention_layers = num_attention_layers

        # Model initialization.
        self.model = PoseFormer(self.d_pose, self.d_hidden, self.num_attention_layers, self.num_attention_heads,
                                self.d_hidden * 2, dropout=self.pf_dropout, num_classes=self.num_classes,
                                asl_features=kwargs['asl_features'])

        # Metrics.
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
        self.m_accuracy = torchmetrics.Accuracy()

        # Save hyperparameters to model checkpoint.
        self.save_hyperparameters()

    def load_weights(self, checkpoint_path, load_classifier: bool = False):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        if not load_classifier:
            del state_dict['model.head.weight']
            del state_dict['model.head.bias']
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print(f'Loaded checkpoint {checkpoint_path}. The following keys were missing:')
        print(missing_keys)
        print('The following keys were unexpected:')
        print(unexpected_keys)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, targets, _filenames = batch
        z = self.model(inputs)

        loss = self.criterion(z, targets)

        preds = torch.argmax(z, dim=-1)
        batch_size = inputs.size(1)

        self.log('train_loss', loss, batch_size=batch_size)
        self.log('train_accuracy', self.m_accuracy(preds, targets), batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, _filenames = batch
        z = self.model(inputs)

        loss = self.criterion(z, targets)
        preds = torch.argmax(z, dim=-1)

        batch_size = inputs.size(1)
        self.log('val_loss', loss, batch_size=batch_size)
        self.log('val_accuracy', self.m_accuracy(preds, targets), batch_size=batch_size)

        # hp_metric for hyperparameter influence on accuracy tracking in tensorboard.
        self.log('hp_metric', self.m_accuracy(preds, targets), batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if not self.retrain_on_all:
            return {
                'optimizer': optimizer,
                'lr_scheduler': ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5),
                'monitor': 'val_accuracy'
            }
        else:
            return {
                'optimizer': optimizer,
                # When retrain-on-all is enabled, we use MultiStepLR to mimic ReduceLROnPlateau.
                'lr_scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, self.lr_steps, gamma=0.1),
            }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        # Module.
        parser.add_argument('--batch_size', type=int, help='Batch size.', required=True)
        parser.add_argument('--learning_rate', type=float, help='Base learning rate.', required=True)
        parser.add_argument('--weight_decay', type=float, help='Optimizer weight decay.')
        parser.add_argument('--num_classes', type=int, help='Number of classes.', required=True)
        parser.add_argument('--num_attention_layers', type=int, help='Number of multi-head attention layers.',
                            required=True)
        parser.add_argument('--num_attention_heads', type=int, help='Number of attention heads per layer.',
                            required=True)
        parser.add_argument('--pf_dropout', type=float, default=0.2, help='PoseFormer dropout.')
        parser.add_argument('--d_hidden', type=int, help='Dimensionality of attention layers.', required=True)
        parser.add_argument('--d_pose', type=int, help='Number of input features for pose data.', required=True)
        parser.add_argument('--asl_features', action='store_true', help='Enable this for the Kaggle model.')

        return parent_parser
