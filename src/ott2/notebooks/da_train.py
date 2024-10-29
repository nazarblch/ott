import sys , os
# import rootutils

# ROOT = rootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator='pyproject.toml')
sys.path.append("/home/nazar/projects/ott5/src")
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
torch.backends.cudnn.deterministic = True


from ott2.notebooks.tools import calculate_frechet_distance, get_loader_stats, get_pushed_loader_stats, load_dataset
from typing import Iterator, Literal, NamedTuple, Optional, Tuple
import dataclasses
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from ott2.neural.methods.enot_nnx import ExpectileNeuralDual
from ott2.geometry import costs
from ott2.notebooks.resnet import ResNet_D
from ott2.notebooks.unet_nnx import UNet
import optax
import os
import gc
import jax_inception as inception
import functools
from flax import nnx

print(jax.devices())

image_size = 64
batch_size = 64
# DATASET_12_LIST = ('handbag', f'/home/nazar/projects/handbag_{image_size}.hdf5', 
#                    'shoes', f'/home/nazar/projects/shoes_{image_size}.hdf5')

# DATASET_12_LIST = ('ffhq_faces', '/home/jovyan/nazar/ffhq_faces/', 
#                     'comic_faces', '/home/jovyan/nazar/comic_faces_v2/')

DATASET_12_LIST = ('celeba_female', '/home/nazar/projects/celeba_female/', 
                   'aligned_anime_faces', '/home/nazar/projects/aligned_anime_faces/')


class Dataset(NamedTuple):
  source_iter: Iterator[jnp.ndarray]
  target_iter: Iterator[jnp.ndarray]


@dataclasses.dataclass
class ImagesSampler:
    def __init__(self, sampler, size, img_size):
        self.sampler = sampler
        self.size = size
        self.img_size = img_size
        
    def __iter__(self):
        rng = jax.random.PRNGKey(0)
        while True:
            rng, sample_key = jax.random.split(rng, 2)
            yield self._sample(sample_key, self.size)
            
    def _sample(self, key, batch_size):
        return jnp.asarray(self.sampler.sample(batch_size).numpy()).reshape(batch_size, 3 * self.img_size * self.img_size)
    

train_source, test_source = load_dataset(DATASET_12_LIST[0], DATASET_12_LIST[1], img_size=image_size, batch_size=batch_size, device="cpu")
gc.collect()
train_target, test_target = load_dataset(DATASET_12_LIST[2], DATASET_12_LIST[3], img_size=image_size, batch_size=batch_size, device="cpu")
gc.collect()

train_loader = Dataset(
    source_iter=iter(ImagesSampler(train_source, batch_size, image_size)),
    target_iter=iter(ImagesSampler(train_target, batch_size, image_size))
)
valid_loader = train_loader


eval_data_source = next(valid_loader.source_iter)
eval_data_target = next(valid_loader.target_iter)

def to_img(x):
    x = x.reshape(-1, 3, image_size, image_size).transpose(0, 2, 3, 1)
    x = (x * 0.5 + 0.5).clip(0., 1.)
    return x


inception_net = inception.InceptionV3(pretrained=True)
rng = jax.random.PRNGKey(0)
inception_params = inception_net.init(rng, jnp.ones((1, 299, 299, 3)))
inception_apply = nnx.jit(functools.partial(inception_net.apply, train=False))

mu_data, sigma_data = get_loader_stats(test_target.loader, inception_apply, inception_params, batch_size=128, n_epochs=1, verbose=True)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("/home/nazar/pomoika/enot_0.98")


def training_callback(step, learned_potentials):
    if step % 1000 == 10:
        # clear_output()
        print(f"Training iteration: {step}")
        neural_dual_dist = learned_potentials.distance(
            eval_data_source, eval_data_target
        )
        print(
            f"Neural dual distance between source and target data: {neural_dual_dist:.5f}"
        )
        pred = learned_potentials.transport(eval_data_source)

        fig, ax = plt.subplots(6, 2)

        for k in range(6):
            ax[k, 0].imshow(to_img(eval_data_source[k])[0])
            ax[k, 1].imshow(to_img(pred[k])[0])

        fig.savefig(f"/home/nazar/projects/outputs5.5/{step}.png")

        # display(fig)
        plt.close(fig)

        mu, sigma, mse = get_pushed_loader_stats(learned_potentials.transport, test_source.loader, inception_apply, inception_params, batch_size=64, device='cuda', verbose=True, upgrade=False)
        fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)

        print("fid=", fid, "mse=", mse)
        writer.add_scalar("fid", fid, step)
        writer.add_scalar("mse", mse, step)



print("init")

neural_f = UNet(nnx.Rngs(0), image_size, 3, 3, 48 * 2)
neural_g = ResNet_D(image_size, nfilter=48 * 2, nlayers=4)

num_train_iters = 200_100

lr_schedule_f = optax.cosine_decay_schedule(
    init_value=1e-4, decay_steps=num_train_iters, alpha=1e-2
)

lr_schedule_g = optax.cosine_decay_schedule(
    init_value=5e-5, decay_steps=num_train_iters, alpha=1e-2
)

optimizer_f = optax.adamw(learning_rate=lr_schedule_f, b1=0.5, b2=0.5, eps=1e-6)
optimizer_g = optax.adamw(learning_rate=lr_schedule_g, b1=0.5, b2=0.5, eps=1e-6)

print("init ot")


@jax.tree_util.register_pytree_node_class
class MSECost(costs.SqEuclidean):
  
  def norm(self, x: jnp.ndarray):
    """Compute squared Euclidean norm for vector."""
    x = jnp.atleast_2d(x)
    return jnp.sum(x ** 2, axis=-1).squeeze() / (3 * image_size)

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute minus twice the dot-product between vectors."""
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    return -2.0 * jnp.vdot(x, y).squeeze() / (3 * image_size)

  def h(self, z: jnp.ndarray) -> float:  # noqa: D102
    return jnp.sum(z ** 2) / (3 * image_size)

  def h_legendre(self, z: jnp.ndarray) -> float:  # noqa: D102
    return 0.25 * jnp.sum(z ** 2) / (3 * image_size)

import flaxmodels as fm

# vgg16 = fm.VGG16(output='activations', pretrained='imagenet')
# init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
# vgg_params = vgg16.init(init_rngs, jnp.ones((1, 224, 224, 3), dtype=jnp.float32))
# # out = vgg16.apply(params, x, train=False)
# vgg_apply = functools.partial(vgg16.apply, train=False)

# @jax.jit
# def vgg(x):
#      x = x.reshape(-1, 3, image_size, image_size).transpose(0,2,3,1)
#      x = (x * 0.5 + 0.5)
#      x = jax.image.resize(x, (x.shape[0], 224, 224, 3), method='bilinear')
#      out = vgg_apply(vgg_params, x)

#      return (out["relu1_2"].reshape(x.shape[0], -1), 
#              out["relu2_2"].reshape(x.shape[0], -1), 
#              out["relu4_2"].reshape(x.shape[0], -1), 
#              out["relu5_2"].reshape(x.shape[0], -1) ) 


# @jax.jit
# def inception(x):
#     x = x.reshape(-1, 3, image_size, image_size).transpose(0,2,3,1)
#     x = x * 0.5 + 0.5
#     x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear')
#     fx = inception_apply(inception_params, x).reshape(x.shape[0], -1) 
#     return fx 


# @jax.tree_util.register_pytree_node_class
# class MSECost(costs.CostFn):
  

#   def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
#     """Compute minus twice the dot-product between vectors."""
#     x = jnp.atleast_2d(x)
#     y = jnp.atleast_2d(y)

#     # x1, x2, x3, x4 = vgg(x)
#     # y1, y2, y3, y4 = vgg(y)
#     x1 = inception(x)
#     y1 = inception(y)

#     l1 = jnp.abs(x - y).mean(1)
#     l2 = ((x - y)**2).mean(1)
#     # l_vgg = ((x1 - y1)**2).mean(1) + ((x2 - y2)**2).mean(1) + ((x3 - y3)**2).mean(1) + ((x4 - y4)**2).mean(1)
#     l_inc = ((x1 - y1)**2).mean(1)

#     return jnp.squeeze(l2 + l1 * 0.2 + l_inc) * 10

neural_dual_solver = ExpectileNeuralDual(
    3 * image_size * image_size,
    neural_f,
    neural_g,
    optimizer_f,
    optimizer_g,
    cost_fn=MSECost(),
    num_train_iters=num_train_iters,
    expectile=0.98,
    expectile_loss_coef=1.0,
    rng=jax.random.PRNGKey(5),
    is_bidirectional=False,
    # start_step=0
    # use_dot_product=True
)

print("train")

learned_potentials = neural_dual_solver(
    *train_loader,
    *valid_loader,
    callback=training_callback
)