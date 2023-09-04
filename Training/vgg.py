import torch.nn as nn
import torch.nn.functional as F


def vgg_block(num_convs, output_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(output_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.LazyBatchNorm2d())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight)


class VGG(nn.Module):
    def __init__(self, arch, lr=0.1, num_classes=8, drop=0.2):
        super().__init__()
        self.arch = arch
        self.lr = lr
        self.num_classes = num_classes
        self.drop = drop

        conv_blocks = []
        for num_convs, output_channels in arch:
            conv_blocks.append(vgg_block(num_convs, output_channels))

        self.net = nn.Sequential(
            *conv_blocks,
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.LazyLinear(num_classes)
        )
        self.net.apply(init_cnn)

    def forward(self, X):
        return self.net(X)
