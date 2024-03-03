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

class conv_bn_act(nn.Module):
    def __init__(self, in_channel,filters, kernel_size, strides=(1, 1), activation='relu', padding='same'):
        super(conv_bn_act, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv2d(in_channel, filters, kernel_size=kernel_size, stride=strides, padding=padding)
        self.bn_act = bn_act(filters, activation=activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn_act(x)
        return x

def merge(l, filters=None):
    if filters is None:
        filters = l[0].shape[1]  
    x = torch.add(l[0], l[1])
    return x