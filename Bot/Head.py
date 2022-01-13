from collections import OrderedDict

from torch import nn


class Policy(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()

        self.blocks = nn.Sequential(OrderedDict({
            'policy_conv1': nn.Conv2d(in_channels, in_channels, (4, 4), stride=(1, 1), padding='same', *args, **kwargs),
            'policy_fc': nn.Linear(in_channels, out_channels)
        }))

    def forward(self, x):
        x = self.blocks(x)
        return x


class Value(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()

        self.blocks = nn.Sequential(OrderedDict({
            'value_conv1': nn.Conv2d(in_channels, in_channels/2, (4, 4), stride=(1, 1), padding='same', *args, **kwargs),
            'value_conv2': nn.Conv2d(in_channels, 128, (4, 4), stride=(1, 1), padding='same', *args, **kwargs),
            'value_ReLU': nn.ReLU(),
            'value_fc': nn.Linear(in_channels, out_channels)
        }))

    def forward(self, x):
        x = self.blocks(x)
        return x
