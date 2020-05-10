__author__ = 'Michael Guarino (mguarin0)'


import os
import pprint
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models

from ignite.engine import Engine, Events
from ignite.metrics import (Accuracy, Loss,
                            RunningAverage,
                            Precision, Recall)
from ignite.handlers import (Checkpoint, DiskSaver,
                             global_step_from_engine)
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *

from advertorch.context import ctx_noparamgrad_and_eval

from pytorch_memlab import MemReporter

from utils import (create_summary_writer,
                   log_results,
                   print_model)


def run_trainer(data_loader: dict,
        model: models,
        optimizer: optim,
        lr_scheduler: optim.lr_scheduler,
        attackers: dict,
        criterion: nn,
        train_epochs: int,
        log_training_progress_every: int,
        log_val_progress_every: int,
        checkpoint_every: int,
        tb_summaries_dir: str,
        chkpt_dir: str,
        resume_from: str,
        to_device: object,
        to_cpu: object,
        *args,
        **kwargs
):


  def mk_lr_step(loss):
    lr_scheduler.step(loss)


  def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = map(lambda _: to_device(_), batch)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


  def eval_step(engine, batch):
    model.eval()
    with torch.no_grad():
      x, y = map(lambda _: to_device(_), batch)
      y_pred = model(x)
      return y_pred, y


  def chkpt_score_func(engine):
    val_eval.run(data_loader['val'])
    y_pred, y = val_eval.state.output
    loss = criterion(y_pred, y)
    return np.mean(to_cpu(loss, convert_to_np=True))


  # set up ignite engines 
  trainer = Engine(train_step)
  train_eval = Engine(eval_step)
  val_eval = Engine(eval_step)


  @trainer.on(Events.ITERATION_COMPLETED(every=log_training_progress_every))
  def log_training_results(engine):
    step = True
    run_type = 'train'
    train_eval.run(data_loader['train'])
    y_pred, y = train_eval.state.output
    loss = criterion(y_pred, y)
    log_results(to_cpu(y_pred, convert_to_np=True),
                to_cpu(y, convert_to_np=True),
                to_cpu(loss, convert_to_np=True),
                run_type,
                step,
                engine.state.iteration,
                total_train_steps,
                writer)


  @trainer.on(Events.ITERATION_COMPLETED(every=log_val_progress_every))
  def log_val_results(engine):
    step = True
    run_type = 'val'
    val_eval.run(data_loader['val'])
    y_pred, y = val_eval.state.output
    loss = criterion(y_pred, y)
    mk_lr_step(loss)
    log_results(to_cpu(y_pred, convert_to_np=True),
                to_cpu(y, convert_to_np=True),
                to_cpu(loss, convert_to_np=True),
                run_type,
                step,
                engine.state.iteration,
                total_train_steps,
                writer)

  # set up vars
  total_train_steps = len(data_loader['train'])*train_epochs

  # reporter to identify memory usage
  # bottlenecks throughout network
  reporter = MemReporter()
  print_model(model, reporter)

  # set up tensorboard summary writer
  writer = create_summary_writer(model,
                                 data_loader['train'],
                                 tb_summaries_dir)
  # move model to device
  model = to_device(model)

  # set up progress bar
  RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
  pbar = ProgressBar(persist=True, bar_format="")
  pbar.attach(trainer, ['loss'])

  # set up checkpoint
  objects_to_checkpoint = {'trainer': trainer,
               'model': model,
               'optimizer': optimizer,
               'lr_scheduler': lr_scheduler}
  training_checkpoint = Checkpoint(to_save=objects_to_checkpoint,
                   save_handler=DiskSaver(chkpt_dir, require_empty=False),
                   n_saved=3, filename_prefix='best',
                   score_function=chkpt_score_func,
                   score_name='val_loss')
  
  # register events
  trainer.add_event_handler(Events.ITERATION_COMPLETED(every=checkpoint_every),
                            training_checkpoint)
  
  # if resuming
  if resume_from and os.path.exists(resume_from):
    print(f'resume model from: {resume_from}')
    checkpoint = torch.load(resume_from)
    Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)


  # fire training engine
  trainer.run(data_loader['train'], max_epochs=train_epochs)