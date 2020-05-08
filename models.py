import torch.nn as nn
import torchvision.models as models

def get_torchvision_models(model_type: str,
                           num_classes: int, *args, **kwargs):
  """
  select model type from torchvision or 
  modify this to add your model
  Parameters
  ----------
  model_type : str
      [description]
  num_classes : int
      [description]
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
  # TODO confirm that final activation and loss working properly
  model.fc = nn.Sequential(nn.Linear(num_output, num_classes),
               nn.LogSoftmax())

  return model