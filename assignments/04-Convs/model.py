import torch

from torch import nn


def _conv_block(in_chn, out_chn):
    """Convolution block"""
    return [
        nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    ]


class Model(nn.Module):
    """
    basic CNN model
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """Initialize model"""
        super(Model, self).__init__()
        clf_block = [nn.Flatten(1), nn.Linear(2048, num_classes)]
        self.net = nn.Sequential(
            *_conv_block(num_channels, 32),
            nn.MaxPool2d(4),
            *clf_block,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.net(x)
