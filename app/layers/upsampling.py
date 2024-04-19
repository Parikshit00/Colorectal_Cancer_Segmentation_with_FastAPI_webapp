import torch
import torch.nn as nn
import torch.nn.functional as F

class bn_act(nn.Module):
    def __init__(self, channels, activation='swish'):
        super(bn_act, self).__init__()

        self.batch_norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU() if activation == 'relu' else nn.SiLU()

    def forward(self, x):
        x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class decode(nn.Module):
    def __init__(self, in_channel,filters, scale=2, activation='relu'):
        super(decode, self).__init__()
        self.filters = filters
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=filters, kernel_size=1,padding="same", bias=False)
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=filters, kernel_size=3, padding="same", bias=False)
        self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
        self.conv3 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding="same", bias=False)
        nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        self.conv4 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, padding="same", bias=False)
        nn.init.kaiming_uniform_(self.conv4.weight, mode='fan_in', nonlinearity='relu')

        self.batch_norm_activation = bn_act(filters, activation=activation)

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x))
        merge = torch.add(x1, x2)
        x = self.upsample(merge)
        skip_feature = self.activation(self.conv3(merge))
        skip_feature = self.activation(self.conv4(skip_feature))
        merge = torch.add(merge, skip_feature)
        x = self.batch_norm_activation(x)
        return x