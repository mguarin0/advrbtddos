__author__ = 'Michael Guarino (mguarin0)'
__version__ = '1.0'


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

from pytorch_memlab import MemReporter

from utils import create_summary_writer, calc_metrics_log_summaries, tb_add_scalar, print_model


def add_cli_args(parser: ArgumentParser):
    train_periodic_ops_grp = parser.add_argument_group('train_periodic_ops',
                                                       'arguments for configurating '
                                                       'length of training and periodic '
                                                       'operations.')
    train_periodic_ops_grp.add_argument('--train_epochs',
                                        type=int,
                                        default=20,
                                        help='number of epochs to train (default: 10)')
    train_periodic_ops_grp.add_argument('--log_training_progress_every',
                                        type=int,
                                        default=10,
                                        help='how many batches to wait before logging training status')
    train_periodic_ops_grp.add_argument('--checkpoint_every',
                                        type=int,
                                        default=4096,
                                        help='how many batches to wait before checkpointing')
    batch_config_grp = parser.add_argument_group('batch_configs',
                                                 'configuration of batch dims.')
    batch_config_grp.add_argument('--train_batch_size',
                                  type=int,
                                  default=64,
                                  help='input batch size for training (default: 64)')
    batch_config_grp.add_argument('--val_batch_size',
                                  type=int,
                                  default=64,
                                  help='input batch size for validation (default: 64)')
    batch_config_grp.add_argument('--shuffle',
                                  type=bool,
                                  default=True,
                                  help='shuffle dataset (default: True)')
    resource_paths_grp = parser.add_argument_group('resource_paths',
                                                   'paths to resources.')
    resource_paths_grp.add_argument('--tb_summaries_dir',
                                    type=str,
                                    default='tensorboard_logs',
                                    help='log directory for Tensorboard log output')
    resource_paths_grp.add_argument('--chkpt_dir',
                                    type=str,
                                    default='checkpoints',
                                    help='log directory for Tensorboard log output')
    resource_paths_grp.add_argument('--dataset_root',
                                    type=str,
                                    help='dataset root')
    resource_paths_grp.add_argument('--resume_from',
                                    type=str,
                                    default=None,
                                    help='Path to the checkpoint '
                                         '`.pth` file to resume training from')
    network_dims_grp = parser.add_argument_group('network_dims',
                                                 'dimensions of network.')
    network_dims_grp.add_argument('--num_classes',
                                  type=int,
                                  default=2, # 2 I assume
                                  help='number classes')
    hyperparams_grp = parser.add_argument_group('hyperparams',
                                                'hyperparams for network and '
                                                'optim strategy.')
    hyperparams_grp.add_argument('--lr',
                                 type=float,
                                 default=1e-4,
                                 help='learning rate')
    hyperparams_grp.add_argument('--eps',
                                 type=float,
                                 default=1e-8,
                                 help='term added to denominator to improve numerical stability')
    hyperparams_grp.add_argument('--factor',
                                 type=float,
                                 default=0.1,
                                 help='Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1')
    hyperparams_grp.add_argument('--patience',
                                 type=int,
                                 default=10,
                                 help='Number of epochs with no improvement after which learning rate will be reduced. '
                                      'For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, '
                                      'and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10. ')
    hyperparams_grp.add_argument('--verbose',
                                 type=bool,
                                 default=False,
                                 help='If True, prints a message to stdout for each update. Default: False.')
    hyperparams_grp.add_argument('--threshold',
                                 type=float,
                                 default=1e-4,
                                 help='Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.')
    hyperparams_grp.add_argument('--threshold_mode',
                                 type=str,
                                 default='rel',
                                 choices=['rel', 'max'],
                                 help='One of rel, abs. In rel mode, dynamic_threshold '
                                      '= best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) '
                                      'in min mode. In abs mode, dynamic_threshold = best + threshold in max '
                                      'mode or best - threshold in min mode. Default: ‘rel’.')
    hyperparams_grp.add_argument('--cooldown',
                                 type=int,
                                 default=0,
                                 help='Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.')
    hyperparams_grp.add_argument('--min_lr',
                                 type=float,
                                 default=0,
                                 help='A scalar or a list of scalars. A lower bound on the learning '
                                      'rate of all param groups or each group respectively. Default: 0. ')
    hyperparams_grp.add_argument('--eps_scheduler',
                                 type=float,
                                 default=1e-8,
                                 help='Minimal decay applied to lr. If the difference between new and old lr '
                                      'is smaller than eps, the update is ignored. Default: 1e-8. ')


def run_trainer(dataset_root: str,
                train_epochs: int,
                log_training_progress_every: int,
                checkpoint_every: int,
                train_batch_size: int,
                val_batch_size: int,
                shuffle: bool,
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

    # transforms, datasets, and loaders
    # TODO must change resize
    train_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    val_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = torchvision.datasets.ImageFolder(root=os.path.join(dataset_root, 'train'),
                                                transform=train_tfms)
    val_ds = torchvision.datasets.ImageFolder(root=os.path.join(dataset_root, 'test'),
                                              transform=val_tfms)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size,
                                               shuffle=True, num_workers=os.cpu_count())
    val_loader = torch.utils.data.DataLoader(val_ds,  batch_size=val_batch_size,
                                             shuffle=True, num_workers=os.cpu_count())

    # model step up
    model = models.resnet34(pretrained=False)
    num_output = model.fc.in_features
    # TODO confirm that final activation and loss working properly
    model.fc = nn.Sequential(nn.Linear(num_output, num_classes),
                             nn.LogSoftmax())

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


if __name__ == '__main__':
    parser = ArgumentParser('train system for household '
                            'level property prediction')
    add_cli_args(parser)
    args = vars(parser.parse_args())

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    args['device'] = device

    run_trainer(**args)
