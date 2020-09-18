import torchvision
from torch import nn
import numpy as np


def resnet18(n_classes: int, device: str):
    resnet = torchvision.models.resnet18(pretrained=True)

    for param in resnet.parameters():
        param.requires_grad = False

    in_features = resnet.fc.in_features

    header = nn.Sequential(
        nn.Dropout(0.95),
        nn.Linear(in_features, 2048),
        nn.ReLU(),
        nn.Dropout(0.95),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, n_classes)
    )

    resnet.fc = header
    resnet = resnet.to(device)

    return resnet


def get_number_of_params(model, trainable=True):
    if trainable:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
