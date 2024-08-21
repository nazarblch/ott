# Copyright (c) Meta Platforms, Inc. and affiliates.

import random

import copy

import jax
import jax.numpy as jnp

import gc

import numpy as np
import numpy.random as npr
import sklearn.datasets

import torch

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from dataclasses import dataclass
from abc import ABC, abstractmethod

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use('bmh')

# import ot_utils as utils
import w2benchmark.map_benchmark as mbm
import w2benchmark.distributions as mbm_dists
from typing import Iterator, Literal, NamedTuple, Optional, Tuple



def transport_solve(X, Y, epsilon=1e-2):
    geom = pointcloud.PointCloud(X, Y)
    ot_prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    return solver(ot_prob)

# Get sampler for the synthetic datasets
def get_sampler(name):
    if name.startswith('gauss'):
        return GaussianMixture(name[6:])
    elif name.startswith('sk'):
        return SklearnSampler(name[3:])
    elif name == 'rings':
        return RingSampler(name)
    elif name == 'maf_moon':
        return MAFMoonSampler(name)


# Compute the UVP for the Wasserstein-2 benchmark
def compute_UVP(benchmark, learned_potentials, reverse, size=2**14, batch_size=1024):
    X = benchmark.input_sampler.sample(size); X.requires_grad_(True)
    Y = benchmark.map_fwd(X, nograd=True); Y.requires_grad_(True)
    assert size % batch_size == 0

    if not reverse:
        X_push = torch.zeros_like(Y)
        Y_inv = torch.zeros_like(X)
        start_idx = 0
        end_idx = start_idx + batch_size
        while end_idx <= size:
            X_push[start_idx:end_idx] = torch.from_numpy(
                np.array(learned_potentials.transport(
                    jnp.array(X[start_idx:end_idx].cpu().detach()), forward=True))).to(X.device)
            Y_inv[start_idx:end_idx] = torch.from_numpy(
                np.array(learned_potentials.transport(
                    jnp.array(Y[start_idx:end_idx].cpu().detach()), forward=False))).to(X.device)
            start_idx += batch_size
            end_idx = start_idx + batch_size

    with torch.no_grad():
        L2_UVP_fwd = 100 * (((Y - X_push) ** 2).sum(dim=1).mean() \
                            / benchmark.output_sampler.var).item()
        L2_UVP_inv = 100 * (((X - Y_inv) ** 2).sum(dim=1).mean() / \
                            benchmark.input_sampler.var).item()

        cos_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / \
            (np.sqrt((2 * benchmark.cost) * \
                        ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
        cos_inv = (((X - Y) * (Y_inv - Y)).sum(dim=1).mean() / \
            (np.sqrt((2 * benchmark.cost) * \
                        ((Y_inv - Y) ** 2).sum(dim=1).mean().item()))).item()

    gc.collect(); torch.cuda.empty_cache()
    return L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv

@dataclass
class PairData(ABC):
    batch_size: int

    def __post_init__(self):
        self.sampler = self.load_samplers()

    @abstractmethod
    def load_samplers(self):
        pass
    
    @abstractmethod
    def plot(self, dual_trainer, loc):
        pass

    def eval_extra(self, dual_trainer):
        return {}

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['X_sampler'], d['Y_sampler']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.X_sampler, self.Y_sampler = self.load_samplers()


@dataclass
class Pair2d(PairData):
    mu: str
    nu: str
    input_dim = 2
    bounds = [-10, 10]

    def load_samplers(self):
        X_sampler = get_sampler(self.mu)
        Y_sampler = get_sampler(self.nu)
        return X_sampler, Y_sampler
    
    def sample_target(self, batch_size):
        key = jax.random.PRNGKey(42)
        return self.sampler.sample_target(key, batch_size)
    
    def sample_source(self, batch_size):
        key = jax.random.PRNGKey(42)
        return self.sampler.sample_source(key, batch_size)

    def plot(self, dual_trainer, loc):
        nrow, ncol = 1, 3
        fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow))
        key = jax.random.PRNGKey(0)
        k1, k2, key = jax.random.split(key, 3)
        n_sample = 512
        X = self.X_sampler.sample(k1, n_sample)
        Y = self.Y_sampler.sample(k2, n_sample)
        X_push = dual_trainer.push(X)
        Y_push = dual_trainer.push_inv(Y)

        def plot_lines(A, B):
            xs = np.vstack((A[:,0], B[:,0]))
            ys = np.vstack((A[:,1], B[:,1]))
            ax.plot(xs, ys, color=[0.5, 0.5, 1], alpha=0.1)

        ax = axs[0]
        s = 5
        ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3')
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B')
        ax.scatter(X_push[:,0], X_push[:,1], s=s, color='#F2545B')
        plot_lines(X, X_push)

        ax = axs[1]
        ax.scatter(X[:,0], X[:,1], s=s, color='#1A254B')
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#A7BED3')
        ax.scatter(Y_push[:,0], Y_push[:,1], s=s, color='#F2545B')
        plot_lines(Y, Y_push)

        ax = axs[2]
        s = 5
        ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3', zorder=10)
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B', zorder=10)

        n = 300
        all_data = jnp.concatenate((X.ravel(), Y.ravel()))
        b = jnp.abs(all_data).max().item() * 1.2
        x1 = np.linspace(-b, b, n)
        x2 = np.linspace(-b, b, n)
        X1, X2 = np.meshgrid(x1, x2)
        X1flat = np.ravel(X1)
        X2flat = np.ravel(X2)
        X12flat = np.stack((X1flat, X2flat)).T
        Zflat = utils.vmap_apply(
            dual_trainer.D, dual_trainer.D_params, X12flat)
        Z = np.array(Zflat.reshape(X1.shape))

        CS = ax.contourf(X1, X2, Z, cmap='Blues')

        fig.colorbar(CS, ax=ax)

        fig.tight_layout()
        fig.savefig(loc)
        plt.close(fig)


    def eval_extra(self, dual_trainer):
        key = jax.random.PRNGKey(0)
        k1, k2, key = jax.random.split(key, 3)
        n_sample = 512
        X = self.X_sampler.sample(k1, n_sample)
        Y = self.Y_sampler.sample(k2, n_sample)
        Y_hat = dual_trainer.push(X)
        X_hat = dual_trainer.push_inv(Y)
        X_sinkhorn_out = transport_solve(X, X_hat, epsilon=1e-2)
        Y_sinkhorn_out = transport_solve(Y, Y_hat, epsilon=1e-2)
        inv_error = X_sinkhorn_out.reg_ot_cost
        fwd_error = Y_sinkhorn_out.reg_ot_cost
        print(f'+ fwd_error: {fwd_error:.2f} inv_error: {inv_error:.2f}')
        return {'eval_inv_error': inv_error, 'eval_fwd_error': fwd_error}


@dataclass
class PairImages(PairData):
    mu: str
    nu: str
    reverse: bool
    scale: float
    input_dim = 2
    bounds = [0, 10]

    def load_samplers(self):
        if not hasattr(self, 'reverse'):
            self.reverse = False

        if self.reverse:
            X_sampler = ImageSampler(self.nu, scale=self.scale)
            Y_sampler = ImageSampler(self.mu, scale=self.scale)
        else:
            X_sampler = ImageSampler(self.mu, scale=self.scale)
            Y_sampler = ImageSampler(self.nu, scale=self.scale)
        return X_sampler, Y_sampler

    def plot(self, dual_trainer, loc):
        nrow, ncol = 1, 3
        fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow))
        key = jax.random.PRNGKey(0)
        k1, k2, key = jax.random.split(key, 3)
        n_sample = 512
        X = self.X_sampler.sample(k1, n_sample)
        Y = self.Y_sampler.sample(k2, n_sample)
        X_push = dual_trainer.push(X)
        Y_push = dual_trainer.push_inv(Y)

        def plot_lines(A, B):
            xs = np.vstack((A[:,0], B[:,0]))
            ys = np.vstack((A[:,1], B[:,1]))
            ax.plot(xs, ys, color=[0.5, 0.5, 1], alpha=0.1)

        ax = axs[0]
        s = 5
        ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3')
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B')
        ax.scatter(X_push[:,0], X_push[:,1], s=s, color='#F2545B')
        plot_lines(X, X_push)

        ax = axs[1]
        ax.scatter(X[:,0], X[:,1], s=s, color='#1A254B')
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#A7BED3')
        ax.scatter(Y_push[:,0], Y_push[:,1], s=s, color='#F2545B')
        plot_lines(Y, Y_push)

        ax = axs[2]
        s = 5
        ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3', zorder=10)
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B', zorder=10)

        n = 300
        all_data = jnp.concatenate((X.ravel(), Y.ravel()))
        b = jnp.abs(all_data).max().item() * 1.2
        x1 = np.linspace(-b, b, n)
        x2 = np.linspace(-b, b, n)
        X1, X2 = np.meshgrid(x1, x2)
        X1flat = np.ravel(X1)
        X2flat = np.ravel(X2)
        X12flat = np.stack((X1flat, X2flat)).T
        Zflat = utils.vmap_apply(
            dual_trainer.D, dual_trainer.D_params, X12flat)
        Z = np.array(Zflat.reshape(X1.shape))

        CS = ax.contourf(X1, X2, Z, cmap='Blues')

        fig.colorbar(CS, ax=ax)

        fig.tight_layout()
        fig.savefig(loc)
        plt.close(fig)


    def eval_extra(self, dual_trainer):
        key = jax.random.PRNGKey(0)
        k1, k2, key = jax.random.split(key, 3)
        n_sample = 512
        X = self.X_sampler.sample(k1, n_sample)
        Y = self.Y_sampler.sample(k2, n_sample)
        Y_hat = dual_trainer.push(X)
        X_hat = dual_trainer.push_inv(Y)
        X_sinkhorn_out = transport_solve(X, X_hat, epsilon=1e-2)
        Y_sinkhorn_out = transport_solve(Y, Y_hat, epsilon=1e-2)
        inv_error = X_sinkhorn_out.solver_output.reg_ot_cost
        fwd_error = Y_sinkhorn_out.solver_output.reg_ot_cost
        print(f'+ fwd_error: {fwd_error:.2f} inv_error: {inv_error:.2f}')
        return {'eval_inv_error': inv_error, 'eval_fwd_error': fwd_error}



@dataclass
class GaussianMixture:
    mode: str

    def sample(self, key, batch_size):
        if self.mode == "1_unit":
            centers = jnp.array([[0, 0]])
            scale = 0.
            variance = 1.
        elif self.mode == "1_small":
            centers = jnp.array([[0, 0]])
            scale, variance = 5.0, 0.5
        elif self.mode == "8":
            centers = jnp.array(
                [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)),
                    (1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)),
                    (-1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)),
                    (-1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)),
                ]
            )
            scale, variance = 5.0, 0.5
        elif self.mode == "5_sq":
            centers = jnp.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]])
            scale, variance = 5.0, 0.5
        elif self.mode == "4_sq":
            centers = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
            scale, variance = 5.0, 0.5
        else:
            raise NotImplementedError()

        centers = scale * centers
        k1, k2 = jax.random.split(key, 2)
        sample_centers = jax.random.choice(k1, centers, [batch_size])
        samples = sample_centers + variance**2 * jax.random.normal(k2, [batch_size, 2])
        return samples


@dataclass
class SklearnSampler:
    name: str

    def sample(self, key, batch_size, scale=5.0, variance=0.5):
        seed = jax.random.randint(key, [], minval=0, maxval=1e5).item()
        if self.name == "moon_upper":
            samples, _ = sklearn.datasets.make_moons(n_samples=(batch_size, 0), random_state=seed)
        elif self.name == "moon_lower":
            samples, _ = sklearn.datasets.make_moons(n_samples=(0, batch_size), random_state=seed)
        elif self.name == "circle_small":
            samples, _ = sklearn.datasets.make_circles(n_samples=(0, batch_size),
                                                       factor=.5, noise=0.01, random_state=seed)
        elif self.name == "circle_big":
            samples, _ = sklearn.datasets.make_circles(n_samples=(batch_size, 0),
                                                       factor=.5, noise=0.01, random_state=seed)
        elif self.name == "s_curve":
            scale = 5.
            X, c = sklearn.datasets.make_s_curve(batch_size, noise=0.01, random_state=seed)
            samples = X[:,[2,0]]*scale
        elif self.name == "swiss":
            scale = .5
            X, c = sklearn.datasets.make_swiss_roll(batch_size, noise=0.01, random_state=seed)
            samples = X[:,[2,0]]*scale
        return jnp.array(samples)


@dataclass
class ImageSampler:
    image_path: str
    scale: float

    def __post_init__(self):
        if not os.path.exists(self.image_path):
            # Try making the path relative to the w2ot repo root
            self.image_path = SCRIPT_DIR + '/../' + self.image_path
            assert os.path.exists(self.image_path)

        from skimage.io import imread
        self.img = 1.-imread(self.image_path)/255.
        assert self.img.ndim == 2, "Currently only grayscale images are supported"
        self.img_flat = self.img.reshape(-1)
        self.img_flat = self.img_flat/ self.img_flat.sum()
        # Keep the image on the CPU since it may be big

    def sample(self, key, batch_size):
        # Keep the image on the CPU since it may be big
        indices = npr.choice(a=len(self.img_flat), size=[batch_size], p=self.img_flat)

        # Sampling in jax can run out of GPU memory easily:
        # indices = jax.random.categorical(key, logits=self.img_flat, shape=[batch_size])

        h, w = self.img.shape
        max_dim = max(h,w)
        x1 = indices % w
        x2 = h - (indices // w)
        X = jnp.stack((x1, x2)).T / max_dim
        X = (X*self.scale) - self.scale/2.
        return X.astype("float32")


@dataclass
class RingSampler:
    name: str

    def sample(self, key, batch_size):
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = jnp.linspace(0, 2 * jnp.pi, n_samples4, endpoint=False)
        linspace3 = jnp.linspace(0, 2 * jnp.pi, n_samples3, endpoint=False)
        linspace2 = jnp.linspace(0, 2 * jnp.pi, n_samples2, endpoint=False)
        linspace1 = jnp.linspace(0, 2 * jnp.pi, n_samples1, endpoint=False)

        circ4_x = jnp.cos(linspace4)
        circ4_y = jnp.sin(linspace4)
        circ3_x = jnp.cos(linspace4) * 0.75
        circ3_y = jnp.sin(linspace3) * 0.75
        circ2_x = jnp.cos(linspace2) * 0.5
        circ2_y = jnp.sin(linspace2) * 0.5
        circ1_x = jnp.cos(linspace1) * 0.25
        circ1_y = jnp.sin(linspace1) * 0.25

        X = jnp.vstack([
            jnp.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            jnp.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = sklearn.utils.shuffle(X)

        # Add noise
        X = X + jax.random.normal(key, shape=X.shape)*0.08

        return X.astype("float32")


@dataclass
class MAFMoonSampler:
    name: str

    def sample(self, key, batch_size):
        x = jax.random.normal(key, shape=[batch_size, 2])
        x = x.at[:,0].add(x[:,1] ** 2)
        x = x.at[:,0].mul(0.5)
        x = x.at[:,0].add(-2)
        return x


@dataclass
class BenchmarkHDPair(PairData):
    input_dim: int
    reverse: bool
    benchmark_repo_dir: str
    bounds = [-5, 5]
    batch_size: int
    num_samples_to_viz: int
    
    source_iterator = None
    target_iterator = None
    
    def __post_init__(self):
        self.benchmark = mbm.Mix3ToMix10Benchmark(
            self.benchmark_repo_dir,
            self.input_dim,
        )
        super().__post_init__()
        self.source_iterator = self.sample_source(self.batch_size)
        self.target_iterator = self.sample_target(self.batch_size)

    def load_samplers(self):
        if not self.reverse:
            sampler = MBMSampler(mbm_source_sampler=self.benchmark.input_sampler,
                                   mbm_target_sampler=self.benchmark.output_sampler)
        else:
            sampler = MBMSampler(mbm_source_sampler=self.benchmark.output_sampler,
                                   mbm_target_sampler=self.benchmark.input_sampler)
        return sampler
    
    def sample_target(self, batch_size):
        key = jax.random.PRNGKey(42)
        return self.sampler.sample_target(key, batch_size)
    
    def sample_source(self, batch_size):
        key = jax.random.PRNGKey(42)
        return self.sampler.sample_source(key, batch_size)
    
    
    def plot_measure_samples(self, dataloder, save_to: str = 'log_plots'):
        emb_X = PCA(n_components=2).fit_transform(next(dataloder.source_iter))
        emb_Y = PCA(n_components=2).fit_transform(next(dataloder.target_iter))
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.02, "hspace": 0},
                            sharey=True, sharex=True)
        axs[0].scatter(emb_X[:, 0], emb_X[:, 1], color='blue', s=10, alpha=1, edgecolors='black')
        axs[1].scatter(emb_Y[:, 0], emb_Y[:, 1], color='#90EE90', s=10, alpha=1, edgecolors='black')
        axs[0].set_title(r'Source measure $\mathbb{P}$', fontsize=12)
        axs[1].set_title(r'Target measure $\mathbb{Q}$', fontsize=12)
        
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_xlim(self.bounds[0], self.bounds[1])
            # ax.set_ylim(self.bounds[0], self.bounds[1])
        fig.savefig(fname=save_to + "/measures.png")
        print(f"Saved plot to {save_to} directory")
        plt.close(fig)
    
    def plot(self, dual_trainer, loc):
        n_sample = 512 # from paper viz
        
        nrow, ncol = 1, 2
        fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow))
        
        key = jax.random.PRNGKey(0)
        k1, k2, key = jax.random.split(key, 3)
        X = self.X_sampler.sample(k1, n_sample)
        Y = self.Y_sampler.sample(k2, n_sample)
        X_push = dual_trainer.push(X)
        Y_push = dual_trainer.push_inv(Y)

        def plot_lines(A, B):
            xs = np.vstack((A[:,0], B[:,0]))
            ys = np.vstack((A[:,1], B[:,1]))
            ax.plot(xs, ys, color=[0.5, 0.5, 1], alpha=0.1)

        ax = axs[0]
        s = 5
        ax.scatter(X[:,0], X[:,1], s=s, color='#A7BED3')
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#1A254B')
        ax.scatter(X_push[:,0], X_push[:,1], s=s, color='#F2545B')
        plot_lines(X, X_push)

        ax = axs[1]
        ax.scatter(X[:,0], X[:,1], s=s, color='#1A254B')
        ax.scatter(Y[:,0], Y[:,1], s=s, color='#A7BED3')
        ax.scatter(Y_push[:,0], Y_push[:,1], s=s, color='#F2545B')
        plot_lines(Y, Y_push)

        fig.tight_layout()
        fig.savefig(loc)
        plt.close(fig)


    def eval_extra(self, learned_potentials):
        L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv = compute_UVP(
            self.benchmark, learned_potentials, self.reverse)
        #print(f'+ [fwd] UVP: {L2_UVP_fwd:.2f} cos: {cos_fwd:.2f} [inv] UVP: {L2_UVP_inv:.2f} cos: {cos_inv:.2f}')
        return {'UVP_fwd': L2_UVP_fwd, 'cos_fwd': cos_fwd,
                'UVP_inv': L2_UVP_inv, 'cos_inv': cos_inv}

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['X_sampler'], d['Y_sampler']
        del d['benchmark']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.benchmark = mbm.Mix3ToMix10Benchmark(self.benchmark_repo_dir, self.input_dim)
        self.X_sampler, self.Y_sampler = self.load_samplers()


@dataclass
class BenchmarkImagePair(PairData):
    which: str
    batch_size: int
    reverse: bool
    input_dim = 64*64*3
    bounds = [-1, 1]
    benchmark_repo_dir: str

    def __post_init__(self):
        self.benchmark = mbm.CelebA64Benchmark(
            self.benchmark_repo_dir,
            which=self.which, batch_size=self.batch_size,
            device='cuda')
        super().__post_init__()
        self.source_iterator = self.sample_source(self.batch_size)
        self.target_iterator = self.sample_target(self.batch_size)
        
    def load_samplers(self):
        if not self.reverse:
            sampler = MBMSampler(mbm_source_sampler=self.benchmark.input_sampler,
                                   mbm_target_sampler=self.benchmark.output_sampler)
        else:
            sampler = MBMSampler(mbm_source_sampler=self.benchmark.output_sampler,
                                   mbm_target_sampler=self.benchmark.input_sampler)
        return sampler
    
    def sample_target(self, batch_size):
        key = jax.random.PRNGKey(42)
        return self.sampler.sample_target(key, batch_size)
    
    def sample_source(self, batch_size):
        key = jax.random.PRNGKey(42)
        return self.sampler.sample_source(key, batch_size)

    def plot(self, learned_potentials, loc):
        n_sample = 4

        nrow, ncol = n_sample, 4
        fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow),
                                gridspec_kw = {'wspace':0, 'hspace':0})

        key = jax.random.PRNGKey(0)
        k1, k2, key = jax.random.split(key, 3)
        X = self.X_sampler.sample(k1, n_sample)
        Y = self.Y_sampler.sample(k2, n_sample)
        X_push = learned_potentials.transport(jnp.array(X), forward=True)
        Y_push = learned_potentials.transport(jnp.array(Y), forward=False)

        def to_img(x):
            x = x.reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)
            x = (x + 1.)/2.
            x = x.clip(0., 1.)
            return x

        X, Y, X_push, Y_push = [to_img(z) for z in [X, Y, X_push, Y_push]]

        for i in range(n_sample):
            axs[i,0].imshow(X[i])
            axs[i,1].imshow(X_push[i])
            axs[i,2].imshow(Y[i])
            axs[i,3].imshow(Y_push[i])

        for ax in axs.ravel():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(False)

        fig.tight_layout()
        fig.savefig(loc)
        print(f'saving to {loc}')
        plt.close(fig)


    def eval_extra(self, learned_potentials, size=4096, batch_size=1024):
        L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv = compute_UVP(
            self.benchmark, learned_potentials, self.reverse, size=size, batch_size=batch_size)
        return {'UVP_fwd': L2_UVP_fwd, 'cos_fwd': cos_fwd,
                'UVP_inv': L2_UVP_inv, 'cos_inv': cos_inv}

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['X_sampler'], d['Y_sampler']
        del d['benchmark']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.benchmark = mbm.CelebA64Benchmark(
            which=self.which, batch_size=self.batch_size, device='cuda',
            benchmark_repo_dir=self.benchmark_repo_dir)
        self.X_sampler, self.Y_sampler = self.load_samplers()

@dataclass
class MBMSampler:
    mbm_source_sampler: mbm_dists.Sampler
    mbm_target_sampler: mbm_dists.Sampler
    
    def sample_source(self, key, batch_size):
        while True:
            key, sample_key = jax.random.split(key, 2)
            seed = jax.random.randint(sample_key, [], minval=0, maxval=1e5).item()
            torch.manual_seed(seed)
            np.random.seed(seed)
            source_samples = self.mbm_source_sampler.sample(batch_size).detach().cpu().numpy()
            yield jnp.array(source_samples)
    
    def sample_target(self, key, batch_size):
        while True:
            key, sample_key = jax.random.split(key, 2)
            seed = jax.random.randint(sample_key, [], minval=0, maxval=1e5).item()
            torch.manual_seed(seed)
            np.random.seed(seed)
            target_samples = self.mbm_target_sampler.sample(batch_size).detach().cpu().numpy()
            yield jnp.array(target_samples)