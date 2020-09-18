import torchvision
from torch import nn
import numpy as np
import torch


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


def get_conv_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU()
    )

    return block


def get_fc_block(n_in, n_out):
    block = nn.Sequential(
        nn.Linear(n_in, n_out),
        # nn.BatchNorm1d(n_out),
        nn.ReLU()
    )

    return block


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int, convs: list, fcs: list, inp_shape: list):
        super(SimpleCNN, self).__init__()

        mock_input = torch.randn(2, *inp_shape)
        in_channels = inp_shape[0]
        self.convs = []
        for out_channels in convs:
            conv = get_conv_block(in_channels, out_channels)
            mock_input = conv(mock_input)
            in_channels = out_channels
            self.convs.append(conv)
        self.convs = nn.Sequential(*self.convs)

        a, b, c, d = mock_input.shape
        n_in = b * c * d
        self._after_conv = n_in
        self.fcs = []
        mock_input = mock_input.view(-1, n_in)
        for n_out in fcs:
            fc = get_fc_block(n_in, n_out)
            mock_input = fc(mock_input)
            n_in = n_out
            self.fcs.append(fc)
        self.fcs = nn.Sequential(*self.fcs)

        self.out = nn.Linear(fcs[-1], n_classes)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._after_conv)
        x = self.fcs(x)
        x = self.out(x)

        return x
