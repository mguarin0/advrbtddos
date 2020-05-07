import argparse
import options
from utils import get_device, get_dataloader
from train import run_trainer

if __name__=='__main__':
  # add cli args
  parser = argparse.ArgumentParser()
  options.training_cfgs(parser)
  args = vars(parser.parse_args())

  args['device'] = get_device()
  print(args)

  loaders = {f'{dataset_type}': get_dataloader(dataset_type=dataset_type, **args)
                 for dataset_type in ['test',
                                      'val',
                                      'train']}
  print(loaders)

# run_trainer(**args)