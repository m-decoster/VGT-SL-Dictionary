import argparse
import csv

import pytorch_lightning as pl
import torch

from data.module import DataModule
from models.module import Module


def test(args):
    """Run the model in inference mode on the test set.

    :param args: The command line arguments"""
    # --- Initialization --- #
    module = Module.load_from_checkpoint(args.checkpoint)
    hparams = dict(module.hparams)

    pl.seed_everything(int(hparams['seed']))

    data_module = DataModule(**hparams)
    data_loaders = [
        ('train', data_module.train_dataloader()),
        ('val', data_module.val_dataloader()),
        ('test', data_module.test_dataloader())
    ]

    module = module.eval()
    if torch.cuda.is_available():
        module = module.cuda()

    all_predictions = []
    with torch.no_grad():
        for subset, data_loader in data_loaders:
            for i, batch in enumerate(data_loader):
                model_inputs, targets, filenames, dataset_indices = batch
                if torch.cuda.is_available():
                    model_inputs = model_inputs.to('cuda')
                    targets = targets.to('cuda')
                    dataset_indices = dataset_indices.to('cuda')

                model_outputs = module(model_inputs, dataset_indices)  # Batch, num classes.
                predictions = torch.argmax(model_outputs, dim=-1)
                predicted_probabilities = torch.softmax(model_outputs, dim=-1)

                for sample_index in range(model_outputs.size(0)):
                    all_predictions.append(
                        [filenames[sample_index], subset, targets[sample_index].item(),
                         predictions[sample_index].item(),
                         *predicted_probabilities[sample_index].detach().cpu().numpy().tolist()])

        num_classes = len(all_predictions[0]) - 4  # - 4: filename, subset, ground truth, prediction
        with open(args.output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Path', 'Subset', 'Ground truth', 'Prediction', *[f'p{i}' for i in range(num_classes)]])
            writer.writerows(all_predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str, help='The path to the checkpoint file.')
    parser.add_argument('output_file', type=str,
                        help='The path to the file where predictions will be written.')

    args = parser.parse_args()

    test(args)
