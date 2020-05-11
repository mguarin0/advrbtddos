__author__ = 'Michael Guarino (mguarin0)'

from argparse import ArgumentParser


def base_training_cfgs(parser: ArgumentParser):


  train_periodic_ops_grp = parser.add_argument_group('train_periodic_ops',
     'arguments for configurating '
     'length of training and periodic '
     'operations.')
  train_periodic_ops_grp.add_argument('--train_epochs',
    type=int,
    default=20,
    help='number of epochs to train (default: 20)')
  train_periodic_ops_grp.add_argument('--log_training_progress_every',
    type=int,
    default=128,
    help='how many steps to wait before logging training status')
  train_periodic_ops_grp.add_argument('--log_val_progress_every',
    type=int,
    default=256,
    help='how many steps to wait before logging validation status')
  train_periodic_ops_grp.add_argument('--checkpoint_every',
    type=int,
    default=1024,
    help='how many steps to wait before checkpointing')


  model_cfg_grp = parser.add_argument_group('model_cfg',
     'configurations for model.')
  model_cfg_grp.add_argument('--model_type',
         type=str,
         default='resnet34',
         help='select model type')
  model_cfg_grp.add_argument('--loss_type',
         type=str,
         default='nn_ce',
         help='loss to use')



  cuda_cfg_grp = parser.add_argument_group('model_cfg',
     'configurations for model.')
  cuda_cfg_grp.add_argument('--use_gpu',
         type=bool,
         default=True,
         help='use gpu')
  cuda_cfg_grp.add_argument('--gpu_id',
         type=int,
         default=None,
         help='gpu id to use')


  dataset_cfg_grp = parser.add_argument_group('dataset_cfg',
     'configurations for dataset.')
  dataset_cfg_grp.add_argument('--height',
     type=int,
     default=400,
     help='height of image')
  dataset_cfg_grp.add_argument('--width',
     type=int,
     default=400,
     help='width of image')
  dataset_cfg_grp.add_argument('--mean_ch_1',
     type=float,
     default=0.5,
     help='mean of channel 1')
  dataset_cfg_grp.add_argument('--mean_ch_2',
     type=float,
     default=0.5,
     help='mean of channel 2')
  dataset_cfg_grp.add_argument('--mean_ch_3',
     type=float,
     default=0.5,
     help='mean of channel 3')
  dataset_cfg_grp.add_argument('--std_ch_1',
     type=float,
     default=0.5,
     help='std of channel 1')
  dataset_cfg_grp.add_argument('--std_ch_2',
     type=float,
     default=0.5,
     help='std of channel 2')
  dataset_cfg_grp.add_argument('--std_ch_3',
     type=float,
     default=0.5,
     help='std of channel 3')
  dataset_cfg_grp.add_argument('--rotation_amt',
     type=int,
     default=15,
     help='amount to rotate')
  dataset_cfg_grp.add_argument('--batch_size',
     type=int,
     default=64,
     help='input batch size used for train, val, test (default: 64)')
  dataset_cfg_grp.add_argument('--shuffle',
     type=bool,
     default=True,
     help='shuffle dataset (default: True)')
  dataset_cfg_grp.add_argument('--num_classes',
     type=int,
     default=2,
     help='number classes')



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


def adv_training_cfg(parser: ArgumentParser):


  train_adv_periodic_ops_grp = parser.add_argument_group('train_adv_periodic_ops',
     'arguments for configurating '
     'periodic operations '
     'for adversary.')
  train_adv_periodic_ops_grp.add_argument('--run_adv_every',
    type=int,
    default=3,
    help='run adversary every n steps during training train (default: 3)')


  adv_cfg_grp = parser.add_argument_group('adv_cfg',
     'arguments for configurating '
     'of adversary.')
  adv_cfg_grp.add_argument('--attack_type',
#                          nargs='+',
                           default='linfpgd',
                           choices=['gsa',
                                    'linfpgd', 'singlepixel',
                                    'jacobiansaliencymap'],
                           help='attack type to use during robust training '
                                'multiple attacks can be made (default: linfpgd)')


def adv_attk_params(parser: ArgumentParser):

  adv_gsa_cfg_grp = parser.add_argument_group('adv_gsa_cfg')
  adv_gsa_cfg_grp.add_argument('--adv_gsa_eps', type=float, default=0.3, help='attk step size')
  adv_gsa_cfg_grp.add_argument('--adv_gsa_clip_min', type=float, default=0.0, help='min val')
  adv_gsa_cfg_grp.add_argument('--adv_gsa_clip_max', type=float, default=1.0, help='max val')
  adv_gsa_cfg_grp.add_argument('--adv_gsa_targeted', type=bool, default=False, help='max val')


  adv_linfpgd_cfg_grp = parser.add_argument_group('adv_linfpgd_cfg')
  adv_linfpgd_cfg_grp.add_argument('--adv_linfpgd_eps', type=float, default=0.3, help='max distortion')
  adv_linfpgd_cfg_grp.add_argument('--adv_linfpgd_nb_iter', type=int, default=40, help='max distortion')
  adv_linfpgd_cfg_grp.add_argument('--adv_linfpgd_eps_iter', type=float, default=0.01, help='attack step size')
  adv_linfpgd_cfg_grp.add_argument('--adv_linfpgd_rand_int', type=bool, default=True, help='rand initialize adv')
  adv_linfpgd_cfg_grp.add_argument('--adv_linfpgd_clip_min', type=float, default=0.0, help='min val')
  adv_linfpgd_cfg_grp.add_argument('--adv_linfpgd_clip_max', type=float, default=1.0, help='max val')
  adv_linfpgd_cfg_grp.add_argument('--adv_linfpgd_targeted', type=bool, default=False, help='max val')


  adv_singlepixel_cfg_grp = parser.add_argument_group('adv_singlepixel_cfg')
  adv_singlepixel_cfg_grp.add_argument('--adv_singlepixel_max_pixel', type=int, default=100, help='max number of pixels to perturb')
  adv_singlepixel_cfg_grp.add_argument('--adv_singlepixel_clip_min', type=float, default=0.0, help='min val')
  adv_singlepixel_cfg_grp.add_argument('--adv_singlepixel_clip_max', type=float, default=1.0, help='max val')
  adv_singlepixel_cfg_grp.add_argument('--adv_singlepixel_targeted', type=bool, default=False, help='max val')


  adv_jacobiansaliencymap_cfg_grp = parser.add_argument_group('adv_jacobiansaliencymap_cfg')
  adv_jacobiansaliencymap_cfg_grp.add_argument('--adv_jacobiansaliencymap_num_classes', type=int, default=2, help='number of classes')
  adv_jacobiansaliencymap_cfg_grp.add_argument('--adv_jacobiansaliencymap_clip_min', type=float, default=0.0, help='min val')
  adv_jacobiansaliencymap_cfg_grp.add_argument('--adv_jacobiansaliencymap_clip_max', type=float, default=1.0, help='max val')
  adv_jacobiansaliencymap_cfg_grp.add_argument('--adv_jacobiansaliencymap_gamma', type=float, default=1.0, help='max percent of pixels that can be modified')
  adv_jacobiansaliencymap_cfg_grp.add_argument('--adv_jacobiansaliencymap_theta', type=float, default=1.0, help='perturb length')