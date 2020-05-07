__author__ = 'Michael Guarino (mguarin0)'

from argparse import ArgumentParser

def training_cfgs(parser: ArgumentParser):


  train_periodic_ops_grp = parser.add_argument_group('train_periodic_ops',
                             'arguments for configurating '
                             'length of training and periodic '
                             'operations.')
  train_periodic_ops_grp.add_argument('--train_epochs',
                    type=int,
                    default=50,
                    help='number of epochs to train (default: 50)')
  train_periodic_ops_grp.add_argument('--log_training_progress_every',
                    type=int,
                    default=10,
                    help='how many batches to wait before logging training status')
  train_periodic_ops_grp.add_argument('--checkpoint_every',
                    type=int,
                    default=1024,
                    help='how many batches to wait before checkpointing')



  dataset_paths_grp = parser.add_argument_group('dataset_cfg',
                           'configurations for dataset.')
  dataset_paths_grp.add_argument('--height',
                                 type=int,
                                 default=400,
                                 help='height of image')
  dataset_paths_grp.add_argument('--width',
                                 type=int,
                                 default=400,
                                 help='width of image')
  dataset_paths_grp.add_argument('--mean_ch_1',
                                 type=float,
                                 default=0.5,
                                 help='mean of channel 1')
  dataset_paths_grp.add_argument('--mean_ch_2',
                                 type=float,
                                 default=0.5,
                                 help='mean of channel 2')
  dataset_paths_grp.add_argument('--mean_ch_3',
                                 type=float,
                                 default=0.5,
                                 help='mean of channel 3')
  dataset_paths_grp.add_argument('--std_ch_1',
                                 type=float,
                                 default=0.5,
                                 help='std of channel 1')
  dataset_paths_grp.add_argument('--std_ch_2',
                                 type=float,
                                 default=0.5,
                                 help='std of channel 2')
  dataset_paths_grp.add_argument('--std_ch_3',
                                 type=float,
                                 default=0.5,
                                 help='std of channel 3')
  dataset_paths_grp.add_argument('--rotation_amt',
                                 type=int,
                                 default=15,
                                 help='amount to rotate')
  dataset_paths_grp.add_argument('--batch_size',
                                 type=int,
                                 default=64,
                                 help='input batch size for training (default: 64)')
  dataset_paths_grp.add_argument('--shuffle',
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
                  required=True,
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

def options_gba_gradient_sign_attack(parser):
  parser.add_argument('--predict')