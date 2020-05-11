__author__ = 'Michael Guarino (mguarin0)'


from argparse import ArgumentParser
import os
import pprint
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import LinfPGDAttack


from pytorch_memlab import MemReporter

from utils import create_summary_writer, calc_metrics_log_summaries, tb_add_scalar, print_model


def run_trainer(model: models,
        data_loaders: dict,
        train_epochs: int,
        log_training_progress_every: int,
        checkpoint_every: int,
        tb_summaries_dir: str,
        chkpt_dir: str,
        resume_from: str,
        device: str,
        num_classes: int,
        lr: float,
        eps: float,
        factor: float,
        patience: int,
        verbose: bool,
        threshold: float,
        threshold_mode: str,
        cooldown: int,
        min_lr: float,
        eps_scheduler: float,
        *args,
        **kwargs
        ):
  """
  run training

  Parameters
  ----------
  dataset_root: str
     root directory where dataset is kept
  train_epochs: int
    number of training epochs
  log_training_progress_every: int
    log training progress every n steps
  checkpoint_every: int
    checkpoint model every n steps
  train_batch_size: int
    training batch size
  val_batch_size: int
    validation batch size
  test_batch_size: int
    test batch size
  shuffle: bool
    shuffle incoming batch
  tb_summaries_dir: str
    directory to log summaries
  chkpt_dir: str
    directory to output model checkpoints
  resume_from: str
    resume model from `.pth`
  device: str
    device to run computation on
  num_classes: int
    number of classes
  lr: float
    learning rate
  eps: float
    term added to denominator to improve numerical stability
  factor: float
    Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
  patience: int
    Number of epochs with no improvement after which learning rate will be reduced.
    For example, if patience = 2, then we will ignore the first 2 epochs with no improvement,
    and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
  verbose: bool
    If True, prints a message to stdout for each update. Default: False.
  threshold: float
    Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
  threshold_mode: str
    One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold )
    in min mode. In abs mode, dynamic_threshold = best + threshold in max
    mode or best - threshold in min mode. Default: ‘rel’.
  cooldown: int
    Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
  min_lr: float or list
    A scalar or a list of scalars. A lower bound on the learning
    rate of all param groups or each group respectively. Default: 0.
  eps_scheduler: float
    Minimal decay applied to lr. If the difference between new and old lr
    is smaller than eps, the update is ignored. Default: 1e-8.
  """

  # optimizer and lr scheduler
  optimizer = optim.SGD(model.parameters(), lr=lr)
  lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                            factor=factor, patience=patience,
                            verbose=verbose, threshold=threshold,
                            threshold_mode=threshold_mode,
                            cooldown=cooldown,
                            min_lr=min_lr,
                            eps=eps_scheduler)
  if torch.cuda.device_count() > 1:
    print(f'...using {torch.cuda.device_count()} gpus')
    model = nn.DataParallel(model)

  # reporter to identify memory usage bottlenecks throughout network
  reporter = MemReporter()
  print_model(model, reporter)

  # set up pretty printer
  pp = pprint.PrettyPrinter(indent=4)

  # set up tensorboard summary writer
  writer = create_summary_writer(model, train_loader, tb_summaries_dir)

  # set up ignite
  trainer = create_supervised_trainer(model,
                    optimizer,
                    F.nll_loss,
                    device=device)
  evaluator = create_supervised_evaluator(model,
                      metrics={'nll': Loss(F.nll_loss)},
                      device=device)
  


  def mk_lr_step(loss):
    lr_scheduler.step(loss)


  @trainer.on(Events.ITERATION_COMPLETED(every=log_training_progress_every))
  def log_training_loss(engine):
    print('Epoch[{}/{}] Step[{}/{}] Loss: {:.4f}'
        .format(engine.state.epoch,
            train_epochs,
            engine.state.iteration,
            len(train_loader),
            engine.state.output))
    tb_add_scalar(writer, 'train',
            'loss', engine.state.output,
            engine.state.iteration)


  @trainer.on(Events.EPOCH_COMPLETED)
  def log_training_results(engine):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    output = evaluator.state.output
    avg_nll = metrics['nll']
    y_pred = np.argmax(output[0].to('cpu').numpy(), axis=-1)
    y = output[1].to('cpu').numpy()
    wgt_precision, wgt_recall = calc_metrics_log_summaries(writer, 'train',
                                 engine.state.epoch,
                                 y, y_pred,
                                 avg_nll)
    print('Training Results - Epoch: [{}/{}]  WgtPrecision: {:.4f} WgtRecall: {:.4f} AvgLoss: {:.4f}'
        .format(engine.state.epoch,
            train_epochs,
            wgt_precision,
            wgt_recall,
            avg_nll))


  @trainer.on(Events.EPOCH_COMPLETED)
  def log_validation_results(engine):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    output = evaluator.state.output
    avg_nll = metrics['nll']
    mk_lr_step(avg_nll)
    y_pred = np.argmax(output[0].to('cpu').numpy(), axis=-1)
    y = output[1].to('cpu').numpy()
    wgt_precision, wgt_recall = calc_metrics_log_summaries(writer, 'validation',
                                 engine.state.epoch,
                                 y, y_pred,
                                 avg_nll)
    print('Validation Results - Epoch: [{}/{}]  WgtPrecision: {:.4f} WgtRecall: {:.4f} AvgLoss: {:.4f}'
        .format(engine.state.epoch,
            train_epochs,
            wgt_precision,
            wgt_recall,
            avg_nll))


  # checkpointing related handlers
  def score_function(engine):
    """
    used to poll best performaning model on validation set
    for best model checkpointing
    """
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    output = evaluator.state.output
    avg_nll = metrics['nll']
    y_pred = np.argmax(output[0].to('cpu').numpy(), axis=-1)
    y = output[1].to('cpu').numpy()
    wgt_precision, wgt_recall = calc_metrics_log_summaries(writer, 'validation',
                                 engine.state.epoch,
                                 y, y_pred,
                                 avg_nll)
    # assuming we value precision over recall for our use case
    return wgt_precision

  objects_to_checkpoint = {'trainer': trainer,
               'model': model,
               'optimizer': optimizer,
               'lr_scheduler': lr_scheduler}
  training_checkpoint = Checkpoint(to_save=objects_to_checkpoint,
                   save_handler=DiskSaver(chkpt_dir, require_empty=False),
                   n_saved=3,
                   filename_prefix='best',
                   score_function=score_function,
                   score_name='val_nll')
  trainer.add_event_handler(Events.ITERATION_COMPLETED(every=checkpoint_every),
                training_checkpoint)
  if resume_from and os.path.exists(resume_from):
    print(f'resume model from: {resume_from}')
    checkpoint = torch.load(resume_from)
    Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)

  trainer.run(train_loader, max_epochs=train_epochs)

  # close tb summary writer
  writer.close()