import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding,bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias, padding=padding)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class convformer(nn.Module):
    def __init__(self, input_shape,filters, padding="same"):
        super(convformer, self).__init__()
        self.filters = filters
        self.layer_norm = nn.LayerNorm(input_shape,elementwise_affine=True)
        self.separable_conv = SeparableConv2d(filters, filters, kernel_size=(3, 3), padding=padding, bias=False)
        self.attention = nn.MultiheadAttention(embed_dim=filters, num_heads=1, batch_first=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(filters,filters)
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias,0)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(in_features=filters, out_features=filters)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias,0)

    def _reshape_input(self, x):
        # Reshape x to match the expected shape for nn.MultiheadAttention
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, height * width, channels)
        return x, height

    def _undo_reshape_input(self, x, spat_dim):
        batch_size, seq_len, embed_dim = x.size()
        # Reshape x back to match the original shape
        x = x.reshape(batch_size, embed_dim, spat_dim, spat_dim)
        return x

    def forward(self, input_tensor):
        x = self.layer_norm(input_tensor)
        x = self.separable_conv(x)
        x, spat_dim = self._reshape_input(x)
        x, _ = self.attention(x, x, x)
        x = self._undo_reshape_input(x, spat_dim)
        out = torch.add(x, input_tensor)
        x1 = out.permute(0,2,3,1)
        x1 = self.gelu(self.linear1(x1))
        x1 = self.linear2(x1)
        x1 = x1.permute(0,3,1,2)
        out_tensor = torch.add(out, x1)
        return out