import os, sys

from omegaconf import DictConfig
sys.path.append("..")
sys.path.append("../..")
os.environ['CUDA_VISIBLE_DEVICES']='0'

import matplotlib.pyplot as plt
plt.style.use('bmh')
from verka.data import BenchmarkHDPair

import jax.numpy as jnp
from w2benchmark.metrics import score_baseline_maps, metrics_to_dict
from w2benchmark.plotters import plot_benchmark_metrics

import jax
import optax

import matplotlib.pyplot as plt


from ott import datasets
from ott.datasets import Dataset
from ott.geometry import costs, pointcloud
from ott.neural.methods.expectile_neural_dual import MLP, ExpectileNeuralDual
from ott.tools import sinkhorn_divergence

from typing import Iterator, Literal, NamedTuple, Optional, Tuple


class Dataset(NamedTuple):
  source_iter: Iterator[jnp.ndarray]
  target_iter: Iterator[jnp.ndarray]


benchmark = BenchmarkHDPair(1024, 64, reverse=False, benchmark_repo_dir="/home/m_bobrin/Wasserstein2Benchmark", num_samples_to_viz=512)

train_dataloaders = Dataset(source_iter=benchmark.source_iterator,
                    target_iter=benchmark.target_iterator)
valid_dataloaders = Dataset(source_iter=benchmark.source_iterator,
                    target_iter=benchmark.target_iterator)



num_train_iters = 300_001

neural_f = MLP(dim_hidden=[512, 512, 512, 1], act_fn=jax.nn.elu)
neural_g = MLP(dim_hidden=[512, 512, 512, 1], act_fn=jax.nn.elu)

lr_schedule_f = optax.cosine_decay_schedule(
    init_value=5e-4, decay_steps=num_train_iters, alpha=1e-2
)

lr_schedule_g = optax.cosine_decay_schedule(
    init_value=5e-4, decay_steps=num_train_iters, alpha=1e-2
)
optimizer_f = optax.adam(learning_rate=lr_schedule_f, b1=0.9, b2=0.99)
optimizer_g = optax.adam(learning_rate=lr_schedule_g, b1=0.9, b2=0.99)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("/home/nazar/projects/ott2/tb/w2_64_1.0_10")

def training_callback(step, learned_potentials):
    if step % 5_000 == 0 and step > 0:
        
        uvp_metrics = benchmark.eval_extra(learned_potentials)
      
        print('UVP_fwd', uvp_metrics['UVP_fwd'])
        print('UVP_inv', uvp_metrics['UVP_inv'])
        writer.add_scalar('UVP_fwd', uvp_metrics['UVP_fwd'], step)
        writer.add_scalar('UVP_inv', uvp_metrics['UVP_inv'], step)


neural_dual_solver = ExpectileNeuralDual(
    64,
    neural_f,
    neural_g,
    optimizer_f,
    optimizer_g,
    cost_fn=costs.SqEuclidean(),
    num_train_iters=num_train_iters,
    expectile=1.0,
    expectile_loss_coef=0.5,
    rng=jax.random.PRNGKey(10),
    is_bidirectional=True,
    use_dot_product=True,
)
learned_potentials = neural_dual_solver(
    *train_dataloaders,
    *valid_dataloaders,
    callback=training_callback
)
