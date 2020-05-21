__author__ = 'Michael Guarino (mguarin0)'
__version__ = '1.0'


import os
import yaml
import numpy as np

from typing import List
from tensorboardX import SummaryWriter
from sklearn.metrics import (precision_score,
               recall_score,
               accuracy_score,
               roc_curve,
               auc,
               classification_report)
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import pytorch_memlab


###############
# DEVICE AND
# MODEL LOGGING
###############
def get_device(use_gpu: bool,
         gpu_id:int=None,
         *args, **kwargs):
  if use_gpu and torch.cuda.is_available():
    if gpu_id is not None:
      device = f'cuda:{gpu_id}'
    else:
      device = 'cuda'
  else:
    device = 'cpu'

  return device


def to_device(device: str, t: torch.Tensor, convert_to_np: bool=False):
  if convert_to_np and device == 'cpu':
    return t.to(device).numpy()
  else:
    return t.to(device)


def print_model(model: nn.Module, reporter: pytorch_memlab.MemReporter) -> None:
  """
  print model memory usage by layer
  Parameters
  ----------
  model: nn.Module
  the torch model
  """
  print('=== model definition ===')
  print(model)
  print('=== model memory usage===')
  reporter.report()


###############
# Log Functions
###############

def create_summary_writer(model: nn.Module,
        data_loader: DataLoader,
        tb_summaries_dir: str) -> SummaryWriter:
  """
  create tensorboardX summary writer

  Parameters
  ----------
  model: nn.Module
  pytorch model
  data_loader: DataLoader
  tb_summaries_dir: str

  Returns
  ----------
  tensorboardX summary writer
  """
  writer = SummaryWriter(logdir=tb_summaries_dir)
  data_loader_iter = iter(data_loader)
  x, _ = next(data_loader_iter)
  try:
    writer.add_graph(model, x)
  except Exception as e:
    print("Failed to save model graph: {}".format(e))
  return writer


def log_results(y_pred: np.array,
        y: np.array,
        loss: np.array,
        run_type: str,
        step: bool,
        n: int,
        total_n: int,
        writer: SummaryWriter,
        verbose: bool=False):
  y_pred = np.argmax(y_pred, axis=-1)
  wgt_precision,\
  wgt_recall,\
  accuracy = calc_metrics_log_summaries(writer,
                    run_type,
                    n,
                    y, y_pred,
                    loss)
  msg = 'Results: {}'
  if step:
    msg =  msg + ' - Step: '
  else:
    msg = msg + ' - Epoch: '
  msg = msg + '[{}/{}]  WgtPrecision: {:.4f} WgtRecall: {:.4f} AvgLoss: {:.4f} Accuracy: {:.4f}'

  msg.format(run_type, n, total_n,
       wgt_precision, wgt_recall,
       loss, accuracy)
  if verbose:
    print(msg)


def calc_metrics_log_summaries(writer: SummaryWriter,
                 run_type: str,
                 number: int,
                 ys: np.array,
                 y_preds: np.array,
                 loss: float) -> List[float]:
  """
  calculates performance metrics and writes to tb summary as a scalar
  """
  # calculate metrics
  wgt_precision = _wgt_precision(ys, y_preds)
  wgt_recall = _wgt_recall(ys, y_preds)
  accuracy = _accuracy(ys, y_preds)
  report = _classification_report(ys, y_preds)
  precision_by_class = _extract_from_report(report, 'precision')
  recall_by_class = _extract_from_report(report, 'recall')

  # log metrics
  tb_add_scalar(writer, run_type, 'avg_loss', loss, number)
  tb_add_scalar(writer, run_type, 'wgt_precision', wgt_precision, number)
  tb_add_scalar(writer, run_type, 'wgt_recall', wgt_recall, number)
  tb_add_scalar(writer, run_type, 'accuracy', accuracy, number)
  tb_add_scalars(writer, run_type, 'precision_by_class', precision_by_class, number)
  tb_add_scalars(writer, run_type, 'recall_by_class', recall_by_class, number)

  return [wgt_precision, wgt_recall, accuracy]


def _accuracy(ys: np.array,
              y_preds: np.array) -> float:
  return accuracy_score(ys, y_preds)


def _classification_report(ys: np.array,
                           y_preds: np.array) -> dict:
  return classification_report(ys, y_preds, output_dict=True, zero_division=1)


def _extract_from_report(report: dict, metric: str) -> dict:
  keys = []
  vals = []
  for k in report.keys():
    if k not in ['accuracy', 'macro avg', 'weighted avg']:
      keys.append(k)
      vals.append(report[k][metric])
  return {k: v for k, v in list(zip(keys, vals))}


def tb_add_scalars(writer: SummaryWriter,
           run_type: str,
           metric_name: str,
           metrics: dict,
           number: int):
  writer.add_scalars(f'{run_type}/{metric_name}',
             metrics,
             number)


def tb_add_scalar(writer: SummaryWriter,
          run_type: str,
          metric_name: str,
          metric: str,
          number: int):
  writer.add_scalar(f'{run_type}/{metric_name}',
            metric,
            number)


# DataLoaders
def get_dataloader(dataset_type: str, dataset_root: str,
       height: int, width: int,
       batch_size: int,
       shuffle: bool, num_workers: int=os.cpu_count(),
       *args, **kwargs):
  # transforms, datasets, and loaders
  tfms = transforms.Compose([transforms.Resize((height, width)),
           transforms.ToTensor()])
   # datasets
  img_ds = torchvision.datasets.ImageFolder(root=os.path.join(dataset_root, dataset_type),
            transform=tfms)
  # dataloaders
  loader = torch.utils.data.DataLoader(img_ds, batch_size=batch_size,
           shuffle=shuffle, num_workers=num_workers)
  return loader


def _roc_curve(ys: np.array,
               y_preds: np.array) -> float:
  return precision_score(ys, y_preds, average='weighted', zero_division=1)

def _wgt_precision(ys: np.array,
                   y_preds: np.array) -> float:
  return precision_score(ys, y_preds, average='weighted', zero_division=1)


def _wgt_recall(ys: np.array,
                y_preds: np.array) -> float:
  return recall_score(ys, y_preds, average='weighted', zero_division=1)
