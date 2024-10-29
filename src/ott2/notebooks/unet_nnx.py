import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial
import numpy as np
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

    
class DoubleConv(nnx.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels, rngs: nnx.Rngs):
        mid_channels = out_channels if (mid_channels is None) else mid_channels
        self.conv_1 = nnx.Conv(in_channels, mid_channels, kernel_size=(3, 3), rngs=rngs)
        self.conv_2 = nnx.Conv(mid_channels, out_channels, kernel_size=(3, 3), rngs=rngs)
        self.bn_1 = nnx.BatchNorm(mid_channels, rngs=rngs)
        self.bn_2 = nnx.BatchNorm(out_channels, rngs=rngs)

    def __call__(self, x):
        
        z = self.conv_1(x)
        z = self.bn_1(z)
        z = nnx.relu(z)
        z = self.conv_2(z)
        z = self.bn_2(z)
        z = nnx.relu(z)
   
        return z


class Down(nnx.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, rngs):
        self.double_conv = DoubleConv(in_channels, out_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        z = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        z = self.double_conv(z)
        return z


class Up(nnx.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels, rngs):
        self.double_conv = DoubleConv(in_channels, out_channels, mid_channels, rngs=rngs)

    def __call__(self, x1, x2):

        output_size = x1.shape
        output_size_2 = (output_size[0], output_size[1] * 2, output_size[2] * 2, output_size[3])
        x1 = jax.image.resize(x1, output_size_2, method='bilinear')

        x = jnp.concatenate([x2, x1], axis=3)

        return self.double_conv(x)


class OutConv(nnx.Module):

    def __init__(self, in_channels, out_channels, rngs):
        self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x):
        return self.conv(x)


class UNet(nnx.Module):

    def __init__(self,
                 rngs: nnx.Rngs,
                    image_size: int,
                    n_channels: int = 3,
                    n_classes: int = 3,
                    base_factor: int = 32):
        
        self.image_size = image_size
        
        self.double_conv = DoubleConv(n_channels, base_factor, base_factor, rngs=rngs)
        self.down_1 = Down(base_factor, 2 * base_factor, rngs=rngs)
        self.down_2 = Down(2 * base_factor, 4 * base_factor, rngs=rngs)
        self.down_3 = Down(4 * base_factor, 8 * base_factor, rngs=rngs)
        factor = 2 
        self.down_4 = Down(8 * base_factor, 16 * base_factor // factor, rngs=rngs)
        
        self.up_1 = Up(8 * base_factor + 16 * base_factor // factor, 8 * base_factor // factor, 8 * base_factor, rngs=rngs)
        self.up_2 = Up(4 * base_factor + 8 * base_factor // factor, 4 * base_factor // factor, 4 * base_factor, rngs=rngs)
        self.up_3 = Up(2 * base_factor + 4 * base_factor // factor, 2 * base_factor // factor, 2 * base_factor, rngs=rngs)
        self.up_4 = Up(base_factor + 2 * base_factor // factor, base_factor, base_factor, rngs=rngs)

        self.out_conv = OutConv(base_factor, n_classes, rngs=rngs)

    def __call__(self, x):

        D = self.image_size
        x = x.reshape(-1, 3, D, D).transpose(0, 2, 3, 1)

        x1 = self.double_conv(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)

        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)

        logits = self.out_conv(x)

        return logits.transpose(0, 3, 1, 2).reshape(-1, 3 * D * D)


