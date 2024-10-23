from notebooks.tools import calculate_frechet_distance, get_loader_stats, get_pushed_loader_stats, load_dataset
from typing import Iterator, Literal, NamedTuple, Optional, Tuple
import dataclasses
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from ott2.neural.methods.expectile_neural_dual import ExpectileNeuralDual
from ott2.geometry import costs
from notebooks.resnet import ResNet_D
from notebooks.unet import UNet2, UNet
import optax
import os
import torch
import gc
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="1"

image_size = 128
batch_size = 64
# DATASET_12_LIST = ('handbag', f'/home/nazar/projects/handbag_{image_size}.hdf5', 
#                    'shoes', f'/home/nazar/projects/shoes_{image_size}.hdf5')

DATASET_12_LIST = ('ffhq_faces', '/home/nazar/projects/ffhq_faces/', 
                    'comic_faces', '/home/nazar/projects/comic_faces/')

train_source, test_source = load_dataset(DATASET_12_LIST[0], DATASET_12_LIST[1], img_size=image_size, batch_size=batch_size, device="cpu")
gc.collect()
# train_target, test_target = load_dataset(DATASET_12_LIST[2], DATASET_12_LIST[3], img_size=image_size, batch_size=batch_size, device="cpu")
# gc.collect()

dataset = test_source.loader.dataset

indices = [1, 100, 200, 300, 400]
print(indices)
X = torch.cat([dataset[i][0][None] for i in indices]).numpy()
X = X.reshape((-1, 3 * image_size * image_size))
gc.collect()

neural_f = UNet(image_size, 3, 3, 48)
neural_g = ResNet_D(image_size, nfilter=48, nlayers=5)

num_train_iters = 150_100

lr_schedule_f = optax.cosine_decay_schedule(
    init_value=1e-4, decay_steps=num_train_iters, alpha=1e-2
)

lr_schedule_g = optax.cosine_decay_schedule(
    init_value=5e-5, decay_steps=num_train_iters, alpha=1e-2
)


optimizer_f = optax.adamw(learning_rate=lr_schedule_f, b1=0.5, b2=0.5)
optimizer_g = optax.adamw(learning_rate=lr_schedule_g, b1=0.5, b2=0.5)

print("init ot")


@jax.tree_util.register_pytree_node_class
class MSECost(costs.SqEuclidean):
  
  def norm(self, x: jnp.ndarray):
    """Compute squared Euclidean norm for vector."""
    return jnp.sum(x ** 2, axis=-1) / (3 * image_size)

  def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute minus twice the dot-product between vectors."""
    return -2.0 * jnp.vdot(x, y) / (3 * image_size)

  def h(self, z: jnp.ndarray) -> float:  # noqa: D102
    return jnp.sum(z ** 2) / (3 * image_size)

  def h_legendre(self, z: jnp.ndarray) -> float:  # noqa: D102
    return 0.25 * jnp.sum(z ** 2) / (3 * image_size)


potentials = ExpectileNeuralDual(
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
    start_step=80_003
    # use_dot_product=True
).to_dual_potentials()

T_X = np.asarray(potentials.transport(X))
print(T_X.shape)

def to_img(x):
    x = x.reshape(-1, 3, image_size, image_size).transpose(0, 2, 3, 1)
    x = (x * 0.5 + 0.5).clip(0., 1.)
    return x
  
imgs = to_img(np.concatenate([X, T_X])) 


print(imgs.shape)
fig, axes = plt.subplots(2, len(indices), figsize=(len(indices), 2), dpi=300)
for i, ax in enumerate(axes.flatten()):
    ax.imshow(imgs[i])
    ax.get_xaxis().set_visible(False)
    ax.set_yticks([])


fig.tight_layout(pad=0.001)

fig.savefig(f"/home/nazar/projects/ott/src/notebooks/outputs2/result_90.png")
plt.close(fig)
