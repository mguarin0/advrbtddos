__author__ = 'Michael Guarino (mguarin0)'

import argparse
import options
from utils import get_device, get_dataloader, to_device
from models import (get_torchvision_models,
                    get_optimizer, get_lr_scheduler,
                    get_attackers, get_criterion)
from train_ops import run_trainer
from functools import partial

if __name__=='__main__':

  # add run_type cli args
  parser = argparse.ArgumentParser() 

  # add base cli args
  options.base_training_cfgs(parser)

  # parse all args
  args = vars(parser.parse_args())

  args['device'] = get_device(**args)
  args['data_loader'] = {f'{dataset_type}': get_dataloader(dataset_type=dataset_type, **args)
                            for dataset_type in ['val', 'train']}
  args['model'] = get_torchvision_models(**args)
  args['optimizer'] = get_optimizer(**args)
  args['lr_scheduler'] = get_lr_scheduler(**args)
  args['criterion'] = get_criterion(**args)
  args['to_device'] = partial(to_device, args['device'])
  args['to_cpu'] = partial(to_device, 'cpu')
  run_trainer(**args)