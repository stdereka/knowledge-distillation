import torchvision
from torch import nn


def resnet18(n_classes: int, device):
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
