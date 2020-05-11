from torch import optim
import torch.nn as nn
import torchvision.models as models

def get_torchvision_models(model_type: str,
                           num_classes: int, *args, **kwargs):
  """
  select model type from torchvision or 
  modify this to add your model. Hard coding
  `pretrained=False` because we will never
  use the pretrained model provided by `torchvision`
  for our use case.
  """

  model = None
  if model_type == 'resnet18':
    model = models.resnet18(pretrained=False)
  if model_type == 'resnet34':
    model = models.resnet34(pretrained=False)
  if model_type == 'mobilenet_v2':
    model = models.mobilenet_v2(pretrained=False)
  if model_type == 'inception_v3':
    model = models.inception_v3(pretrained=False)
  if model_type == 'vgg16':
    model= models.vgg16(pretrained=False)
  if model is None:
    raise('no model selected')

  num_output = model.fc.in_features
  model.fc = nn.Sequential(nn.Linear(num_output, num_classes),
               nn.LogSoftmax())

  return model


def get_attackers(attack_type: str, cfg: dict, loss_fn: nn, model: models, *args, **kwargs):
  attackers = {}
  if attack_type=='gsa':
    # https://arxiv.org/abs/1412.652
    from advertorch.attacks import GradientSignAttack 
    adversary = GradientSignAttack(model,
                                   loss_fn=loss_fn,
                                   eps=cfg['adv_gsa_eps'],
                                   clip_min=cfg['adv_gsa_clip_min'],
                                   clip_max=cfg['adv_gsa_clip_max'],
                                   targeted=cfg['adv_gsa_targeted'])
#   attackers['gsa'] = adversary
  if attack_type=='linfpgd':
    from advertorch.attacks import LinfPGDAttack
    adversary = LinfPGDAttack(model,
                   loss_fn=loss_fn,
                   eps=cfg['adv_linfpgd_eps'],
                   nb_iter=cfg['adv_linfpgd_nb_iter'],
                   eps_iter=cfg['adv_linfpgd_eps_iter'],
                   rand_init=cfg['adv_linfpgd_rand_int'],
                   clip_min=cfg['adv_linfpgd_clip_min'],
                   clip_max=cfg['adv_linfpgd_clip_max'],
                   targeted=cfg['adv_linfpgd_targeted'])
#   attackers['linfpgd'] = adversary
  if attack_type=='singlepixel':
    # https://arxiv.org/pdf/1612.06299.pdf
    from advertorch.attacks import SinglePixelAttack
    adversary = SinglePixelAttack(model,
                                  loss_fn=loss_fn,
                                  max_pixels=cfg['adv_singlepixel_max_pixel'],
                                  clip_min=cfg['adv_singlepixel_clip_min'],
                                  clip_max=cfg['adv_singlepixel_clip_max'],
                                  targeted=cfg['adv_singlepixel_targeted'])
#   attackers['singlepixel'] = adversary
  if attack_type=='jacobiansaliencymap':
    #  https://arxiv.org/abs/1511.07528v1
    from advertorch.attacks import JacobianSaliencyMapAttack
    adversary = JacobianSaliencyMapAttack(model,
                                          num_classes=cfg['adv_jacobiansaliencymap_num_classes'],
                                          clip_min=cfg['adv_jacobiansaliencymap_clip_min'],
                                          clip_max=cfg['adv_jacobiansaliencymap_clip_max'],
                                          gamma=cfg['adv_jacobiansaliencymap_gamma'],
                                          theta=cfg['adv_jacobiansaliencymap_theta'])
#   attackers['jacobiansaliencymap'] = adversary
  return adversary 


def get_optimizer(model: models,
                  lr: float, optimizer_type: str='sgd',
                  *args, **kwargs):
  """optimizer type; only using sgd but feel free to extend

  Parameters
  ----------
  optimizer_type : str
  model_params : dict
  lr: float
  """

  optimizer = None
  if optimizer_type == 'sgd':
    optimizer  = optim.SGD(model.parameters(), lr=lr)
  if optimizer is None:
    raise('no optimizer selected')
  return optimizer


def get_lr_scheduler(optimizer: optim,
                     factor: float, patience: int,
                     verbose: bool, threshold: float,
                     threshold_mode: float, cooldown: int,
                     eps_scheduler: float,
                     min_lr: float, eps: float, *args, **kwargs):
  """get learning rate scheduler
  only interested in reduce on plateau
  """
  lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                            factor=factor, patience=patience,
                            verbose=verbose, threshold=threshold,
                            threshold_mode=threshold_mode,
                            cooldown=cooldown,
                            min_lr=min_lr,
                            eps=eps_scheduler)

  return lr_scheduler


def get_criterion(loss_type: str, *args, **kwargs):
  if loss_type=='nn_ce':
    loss = nn.CrossEntropyLoss()
  else:
    raise('no loss specified')
  return loss
