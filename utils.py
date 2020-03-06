__author__ = 'Michael Guarino (mguarin0)'
__version__ = '1.0'


import os
import yaml

from typing import List
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, classification_report

import torch
from torch import nn
from torch.utils.data import DataLoader

import pytorch_memlab


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


def calc_metrics_log_summaries(writer: SummaryWriter,
                               run_type: str,
                               number: int,
                               ys: torch.Tensor,
                               y_preds: torch.Tensor,
                               loss: float) -> List[float]:
    """
    Parameters
    ----------
    writer : SummaryWriter
    run_type : str
    number : int
    ys : torch.Tensor
    y_preds : torch.Tensor
    """
    # calculate metrics
    wgt_precision = _wgt_precision(ys, y_preds)
    wgt_recall = _wgt_recall(ys, y_preds)
    report = _classification_report(ys, y_preds)
    precision_by_class = _extract_from_report(report, 'precision')
    recall_by_class = _extract_from_report(report, 'recall')

    # log metrics
    tb_add_scalar(writer, run_type, 'avg_loss', loss, number)
    tb_add_scalar(writer, run_type, 'wgt_precision', wgt_precision, number)
    tb_add_scalar(writer, run_type, 'wgt_recall', wgt_recall, number)
    tb_add_scalars(writer, run_type, 'precision_by_class', precision_by_class, number)
    tb_add_scalars(writer, run_type, 'recall_by_class', recall_by_class, number)

    return [wgt_precision, wgt_recall]


def _wgt_precision(ys: torch.Tensor,
                   y_preds: torch.Tensor) -> float:
    """
    Parameters
    ----------
    ys : torch.Tensor
    y_preds : torch.Tensor

    Returns
    -------
    float
    """
    return precision_score(ys, y_preds, average='weighted', zero_division=1)


def _wgt_recall(ys: torch.Tensor,
                y_preds: torch.Tensor) -> float:
    """
    Parameters
    ----------
    ys : torch.Tensor
    y_preds : torch.Tensor

    Returns
    -------
    float
    """
    return recall_score(ys, y_preds, average='weighted', zero_division=1)


def _classification_report(ys: torch.Tensor,
                           y_preds: torch.Tensor) -> dict:
    """
    Parameters
    ----------
    ys : torch.Tensor
    y_preds : torch.Tensor

    Returns
    -------
    dict
    """
    return classification_report(ys, y_preds, output_dict=True, zero_division=1)


def _extract_from_report(report: dict, metric: str) -> dict:
    """
    Parameters
    ----------
    report : dict
    metric : str

    Returns
    -------
    dict
    """
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
    """
    Parameters
    ----------
    writer : SummaryWriter
    run_type : str
    metric_name : str
    metrics : dict
    number : int
    """
    writer.add_scalars(f'{run_type}/{metric_name}',
                       metrics,
                       number)


def tb_add_scalar(writer: SummaryWriter,
                  run_type: str,
                  metric_name: str,
                  metric: str,
                  number: int):
    """
    Parameters
    ----------
    writer : SummaryWriter
    run_type : str
    metric_name : str
    metric : str
    number : int
    """
    writer.add_scalar(f'{run_type}/{metric_name}',
                      metric,
                      number)
