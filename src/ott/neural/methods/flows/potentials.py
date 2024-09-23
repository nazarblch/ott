import jax.nn as nn
import jax.numpy as jnp
from flax.struct import PyTreeNode
from dataclasses import dataclass

@dataclass
class BoxPotential:
    xmin: float = -0.5
    xmax: float = 0.5
    ymin: float = -0.5
    ymax: float = 0.5
    temp: float = 0.01
    scale: float = 1

    def __post_init__(self):
        if self.scale > 1:
            self.xmin = self.xmin * self.scale
            self.xmax = self.xmax * self.scale
            self.ymin = self.ymin * self.scale
            self.ymax = self.ymax * self.scale

    def __call__(self, x):
        Ux = (nn.sigmoid((x[0] - self.xmin)/ self.temp) - \
              nn.sigmoid((x[0] - self.xmax)/ self.temp))
        Uy = (nn.sigmoid((x[1] - self.ymin)/ self.temp) - \
              nn.sigmoid((x[1] - self.ymax) / self.temp))
        U = Ux * Uy
        return U
    
class SlitPotential(PyTreeNode):
    xmin: float = -0.1
    xmax: float = 0.1
    ymin: float = -0.25
    ymax: float = 0.25
    temp: float = 0.001

    def __call__(self, x):
        Ux = (nn.sigmoid((x[0] - self.xmin) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax) / self.temp))
        Uy = (nn.sigmoid((x[1] - self.ymin) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax) / self.temp)) - 1.
        U = -Ux * Uy
        return U
    
class BabyMazePotential(PyTreeNode):
    xmin1: float = -0.5
    xmax1: float = -0.3
    ymin1: float = -1.99
    ymax1: float = -0.15
    xmin2: float = 0.3
    xmax2: float = 0.5
    ymin2: float = 0.15
    ymax2: float = 1.99
    M_bounds = (0., 10.)
    temp: float = 0.01

    def __call__(self, x):
        Ux1 = (nn.sigmoid((x[0] - self.xmin1) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax1) / self.temp))
        Ux2 = (nn.sigmoid((x[0] - self.xmin2) / self.temp) - \
                nn.sigmoid((x[0] - self.xmax2) / self.temp))

        Uy1 = (nn.sigmoid((x[1] - self.ymin1) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax1) / self.temp)) - 1.

        Uy2 = (nn.sigmoid((x[1] - self.ymin2) / self.temp) - \
                nn.sigmoid((x[1] - self.ymax2) / self.temp)) - 1.
        U = Ux1 * Uy1 + Ux2 * Uy2
        return -U

class WellPotential(PyTreeNode):
    def __call__(self, x):
        U = jnp.sum(x**2)
        return U

class HillPotential(PyTreeNode):
    scale: float = 1

    def __call__(self, x):
        U = jnp.exp(-jnp.sum((x)**2))
        return U  * self.scale
    
class Cross(PyTreeNode):
    def __call__(self, v):
        d = v.shape[0]
        v1 = jnp.mean(v[:d//2])
        v2 = jnp.mean(v[d//2:])
        return (jnp.abs(jnp.sin(v1) * jnp.sin(v2) * jnp.exp(jnp.abs(15 - jnp.sqrt(jnp.sum(jnp.square(v)))/jnp.pi))) + 1)**0.1
