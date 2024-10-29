import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import linen as fnn
from functools import partial
import numpy as np
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union


ModuleDef = Callable[..., Callable]
InitFn = Callable[[Any, Iterable[int], Any], Any]

    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    out_channels: int
    mid_channels: int

    @nn.compact
    def __call__(self, x, train):
        
        mid_channels = self.out_channels if (self.mid_channels is None) else self.mid_channels
        # mutable = self.is_mutable_collection('batch_stats')
        # scale_init = nn.initializers.ones
        
        z = nn.Conv(mid_channels, kernel_size=(3, 3))(x)
        z = nn.BatchNorm(use_running_average=not train)(z)
        z = nn.relu(z)
        z = nn.Conv(self.out_channels, kernel_size=(3, 3))(z)
        z = nn.BatchNorm(use_running_average=not train)(z)
        z = nn.relu(z)
   
        return z


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    out_channels: int

    @nn.compact
    def __call__(self, x, train):
        z = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        z = DoubleConv(self.out_channels, self.out_channels)(z, train)
        return z


class Up(nn.Module):
    """Upscaling then double conv"""

    out_channels: int
    mid_channels: int

    @nn.compact
    def __call__(self, x1, x2, train):

        output_size = x1.shape
        output_size_2 = (output_size[0], output_size[1] * 2, output_size[2] * 2, output_size[3])
        x1 = jax.image.resize(x1, output_size_2, method='bilinear')
        conv = DoubleConv(self.out_channels, self.mid_channels)
        
        # x1 = nn.ConvTranspose(self.in_channels // 2, kernel_size=(2, 2), strides=(2, 2))(x1)
        # conv = DoubleConv(self.in_channels, self.out_channels, self.out_channels)
    
        x = jnp.concatenate([x2, x1], axis=3)

        return conv(x, train)


class OutConv(nn.Module):

    out_channels: int

    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.out_channels, kernel_size=(1, 1))(x)


class UNet(nn.Module):

    image_size: int
    n_channels: int = 3
    n_classes: int = 3
    base_factor: int = 32 

    @nn.compact
    def __call__(self, x, train=True):

        mutable = self.is_mutable_collection('batch_stats')
        train = mutable
        
        D = self.image_size
        x = x.reshape(-1, 3, D, D).transpose(0, 2, 3, 1)

        base_factor = self.base_factor
        
        x1 = DoubleConv(base_factor, base_factor)(x, train)
        x2 = Down(2 * base_factor)(x1, train)
        x3 = Down(4 * base_factor)(x2, train)
        x4 = Down(8 * base_factor)(x3, train)
        factor = 2 
        x5 = Down(16 * base_factor // factor)(x4, train)

        x = Up(8 * base_factor // factor, 8 * base_factor)(x5, x4, train)
        x = Up(4 * base_factor // factor, 4 * base_factor)(x, x3, train)
        x = Up(2 * base_factor // factor, 2 * base_factor)(x, x2, train)
        x = Up(base_factor, base_factor)(x, x1, train)

        logits = OutConv(self.n_classes)(x)

        return logits.transpose(0, 3, 1, 2).reshape(-1, 3 * D * D)


class Encoder(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, x):
        z1 = fnn.Conv(self.features, kernel_size=(3, 3))(x)
        z1 = fnn.relu(z1)
        z1 = fnn.Conv(self.features, kernel_size=(3, 3))(z1)
        z1 = fnn.BatchNorm(use_running_average=not self.training)(z1)
        z1 = fnn.relu(z1)
        z1_pool = fnn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))

        z2 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z1_pool)
        z2 = fnn.relu(z2)
        z2 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z2)
        z2 = fnn.BatchNorm(use_running_average=not self.training)(z2)
        z2 = fnn.relu(z2)
        z2_pool = fnn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))

        z3 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z2_pool)
        z3 = fnn.relu(z3)
        z3 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z3)
        z3 = fnn.BatchNorm(use_running_average=not self.training)(z3)
        z3 = fnn.relu(z3)
        z3_pool = fnn.max_pool(z3, window_shape=(2, 2), strides=(2, 2))

        z4 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z3_pool)
        z4 = fnn.relu(z4)
        z4 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z4)
        z4 = fnn.BatchNorm(use_running_average=not self.training)(z4)
        z4 = fnn.relu(z4)
        # z4_dropout = fnn.Dropout(0.5, deterministic=False)(z4)
        z4_pool = fnn.max_pool(z4, window_shape=(2, 2), strides=(2, 2))

        z5 = fnn.Conv(self.features * 16, kernel_size=(3, 3))(z4_pool)
        z5 = fnn.relu(z5)
        z5 = fnn.Conv(self.features * 16, kernel_size=(3, 3))(z5)
        z5 = fnn.BatchNorm(use_running_average=not self.training)(z5)
        z5 = fnn.relu(z5)
        # z5_dropout = fnn.Dropout(0.5, deterministic=False)(z5)

        return z1, z2, z3, z4, z5


class Decoder(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, z1, z2, z3, z4_dropout, z5_dropout):
        z6_up = jax.image.resize(z5_dropout, shape=(z5_dropout.shape[0], z5_dropout.shape[1] * 2, z5_dropout.shape[2] * 2, z5_dropout.shape[3]),
                                 method='nearest')
        z6 = fnn.Conv(self.features * 8, kernel_size=(2, 2))(z6_up)
        z6 = fnn.relu(z6)
        z6 = jnp.concatenate([z4_dropout, z6], axis=3)
        z6 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
        z6 = fnn.relu(z6)
        z6 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
        z6 = fnn.BatchNorm(use_running_average=not self.training)(z6)
        z6 = fnn.relu(z6)

        z7_up = jax.image.resize(z6, shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
                                 method='nearest')
        z7 = fnn.Conv(self.features * 4, kernel_size=(2, 2))(z7_up)
        z7 = fnn.relu(z7)
        z7 = jnp.concatenate([z3, z7], axis=3)
        z7 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
        z7 = fnn.relu(z7)
        z7 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
        z7 = fnn.BatchNorm(use_running_average=not self.training)(z7)
        z7 = fnn.relu(z7)

        z8_up = jax.image.resize(z7, shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2] * 2, z7.shape[3]),
                                 method='nearest')
        z8 = fnn.Conv(self.features * 2, kernel_size=(2, 2))(z8_up)
        z8 = fnn.relu(z8)
        z8 = jnp.concatenate([z2, z8], axis=3)
        z8 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
        z8 = fnn.relu(z8)
        z8 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
        z8 = fnn.BatchNorm(use_running_average=not self.training)(z8)
        z8 = fnn.relu(z8)

        z9_up = jax.image.resize(z8, shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2] * 2, z8.shape[3]),
                                 method='nearest')
        z9 = fnn.Conv(self.features, kernel_size=(2, 2))(z9_up)
        z9 = fnn.relu(z9)
        z9 = jnp.concatenate([z1, z9], axis=3)
        z9 = fnn.Conv(self.features, kernel_size=(3, 3))(z9)
        z9 = fnn.relu(z9)
        z9 = fnn.Conv(self.features, kernel_size=(3, 3))(z9)
        z9 = fnn.BatchNorm(use_running_average=not self.training)(z9)
        z9 = fnn.relu(z9)

        y = fnn.Conv(3, kernel_size=(1, 1))(z9)
       
        return y


class UNet2(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, x):
        x = x.reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)
        z1, z2, z3, z4_dropout, z5_dropout = Encoder(self.training)(x)
        y = Decoder(self.training)(z1, z2, z3, z4_dropout, z5_dropout)

        y = y.transpose(0, 3, 1, 2).reshape(-1, 3 * 64 * 64)

        return y
