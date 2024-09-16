# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import math
from ott.geometry import costs
from tqdm.auto import tqdm

# import diffrax
from functools import partial
from flax.training import train_state
from flax import linen as nn
import optax
from ott import utils
from ott.neural.methods.flows import dynamics
from ott.neural.networks import velocity_field
from ott.solvers import utils as solver_utils
from flax.training import train_state

import diffrax

__all__ = ["NeuralOC"]


Callback_t = Callable[[int, ], None]

class TimedX(NamedTuple):
  t: Any
  x: Any


class NeuralOC:
  
  def __init__(
      self,
      input_dim: int,
      value_model: nn.Module,
      optimizer: Optional[optax.GradientTransformation],
      flow: dynamics.LagrangianFlow,
      time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
      key:Optional[jax.Array] = None,
      potential_weight: float = 1.0,
      **kwargs: Any,
  ):
    self.value_model = value_model
    self.flow = flow
    self.time_sampler = time_sampler
   
    key, init_key = jax.random.split(key, 2)
    params = value_model.init(
      init_key, 
      jnp.ones([1, 1]), 
      jnp.ones([1, input_dim]), 
      jnp.ones([1, input_dim])
    )

    self.state = train_state.TrainState.create(
      apply_fn=value_model.apply,
      params=params,
      tx=optimizer
    )

    self.potential_weight = potential_weight
    self.train_step_cost, self.train_step_with_potential = self._get_step_fn()

  def _get_step_fn(self) -> Callable:
      
      def expectile_loss(diff: jnp.ndarray, expectile=0.98) -> jnp.ndarray:
          weight = jnp.where(diff >= 0, expectile, (1 - expectile))
          return weight * diff ** 2

      def am_loss(state, params, key_t, source, target):
        bs = source.shape[0]
        t = self.time_sampler(key_t, bs)
        x_0, x_1 = source, target
        x_t = self.flow.compute_xt(key_t, t, x_0, x_1)
        At_T = self.flow.compute_inverse_control_matrix(t, x_t).transpose()
        U_t = self.flow.compute_potential(t, x_t)

        dsdtdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=[1,2])

        dsdt, dsdx = dsdtdx_fn(params, t, x_t, x_0)
        # keys = jax.random.split(key_t)
        # eps = jax.random.randint(keys[0], x_t.shape, 0, 2).astype(float)*2 - 1.0
        # _, jvp_val = jax.jvp(lambda __x: dsdx_fn(params, t, __x, x_0), (x_t,), (eps,))

        @partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def laplacian(p, t, x, x0):
            fun = lambda __x: state.apply_fn(p,t,__x,x0).sum()
            return jnp.trace(jax.jacfwd(jax.jacrev(fun))(x))

        D = (0.5 * self.flow.compute_sigma_t(t) ** 2).reshape(-1, 1)
        # print(laplacian(params, t, x_t, x_0, key_t).reshape(-1, 1))
        s_diff = dsdt - 0.5 * ((dsdx @ At_T) * dsdx).sum(-1, keepdims=True) + self.potential_weight * U_t + D * laplacian(params, t, x_t, x_0).reshape(-1, 1)
        loss = (s_diff ** 2).mean() + 0.05 * jnp.abs(s_diff).mean() 
        #loss = jnp.abs(s_diff).mean()
        return loss

      def potential_loss(state, params, key, steps_count, weight, source, target):
        bs = source.shape[0]
        # keys = jax.random.split(key)
        t_0, t_1 = jnp.zeros([bs, 1]), jnp.ones([bs, 1])
        x_0, x_1 = source, target
        dt = 1.0 / steps_count

        dsdtdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=[1,2])

        def move(carry, _):
          t_, x_, key_ = carry
          _, dsdx = dsdtdx_fn(state.params, t_, x_, x_0)
          At_T = self.flow.compute_inverse_control_matrix(t_, x_).transpose()
          sigma = self.flow.compute_sigma_t(t_)
          key_, key_s = jax.random.split(key_)
          x_ = x_ - dt * dsdx @ At_T + sigma * jax.random.normal(key_s, shape=x_.shape) * dt
          t_ = t_ + dt
          return (t_, x_, key_), x_
        
        _, result = jax.lax.scan(move, (t_0, x_0, key), None, length=steps_count)
        x_1_pred = jax.lax.stop_gradient(result[-1])

        dual_loss = -(-state.apply_fn(params, t_1, x_1, x_0 * 0) + state.apply_fn(params, t_1, x_1_pred, x_0 * 0)).mean()
        reg_loss = 0

        # exp. reg

        # source, target = x_0, x_1
        # target_hat_detach = x_1_pred
        # batch_cost = lambda x, y: 0.5 * jax.vmap(costs.SqEuclidean())(jnp.atleast_2d(x), jnp.atleast_2d(y)).reshape(-1)
        # g_target = -state.apply_fn(params, t_1, x_1, x_0 * 0).reshape(-1)
        # g_star_source = batch_cost(source, target_hat_detach) + state.apply_fn(params, t_1, x_1_pred, x_0 * 0).reshape(-1)

        # diff_1 = jax.lax.stop_gradient(g_star_source - batch_cost(source, target))\
        #   + g_target
        # reg_loss_1 = expectile_loss(diff_1).mean()

        # diff_2 = jax.lax.stop_gradient(g_target - batch_cost(source, target))\
        #   + g_star_source
        # reg_loss_2 = expectile_loss(diff_2).mean()

        # reg_loss = (reg_loss_1 + reg_loss_2) * 0.5

        return (reg_loss + dual_loss)  * weight

      @jax.jit
      def train_step_cost(state, key, source, target):
        grad_fn = jax.value_and_grad(am_loss, argnums=1, has_aux=False)
        loss, grads = grad_fn(state, state.params, key, source, target)
        state = state.apply_gradients(grads=grads)
        
        return state, loss


      @jax.jit
      def train_step_with_potential(state, key, source, target):
        grad_fn = jax.value_and_grad(am_loss, argnums=1, has_aux=False)
        loss, grads = grad_fn(state, state.params, key, source, target)
        state = state.apply_gradients(grads=grads)
        
        grad_fn = jax.value_and_grad(potential_loss, argnums=1, has_aux=False)
        loss_potential, potential_grads = grad_fn(state, state.params, key, 20, 25, source, target)
        state = state.apply_gradients(grads=potential_grads)
        
        return state, loss, loss_potential
      
      return train_step_cost, train_step_with_potential
  

  def __call__(  # noqa: D102
      self,
      loader: Iterable[Dict[str, np.ndarray]],
      *,
      n_iters: int,
      rng: Optional[jax.Array] = None,
      callback: Optional[Callback_t] = None,
  ) -> Dict[str, List[float]]:
    
    loop_key = utils.default_prng_key(rng)
    training_logs = {"cost_loss": [], "potential_loss": []}

    pbar = tqdm(loader, total=n_iters)
    for it, batch in enumerate(pbar):
      # batch = jtu.tree_map(jnp.asarray, batch)

      src, tgt = batch["src_lin"], batch["tgt_lin"]
      it_key = jax.random.fold_in(loop_key, it)

      if it % 2 != 0:
        self.state, loss = self.train_step_cost(self.state, it_key, src, tgt)
      else:
        self.state, loss, loss_potential = self.train_step_with_potential(self.state, it_key, src, tgt)
        training_logs["potential_loss"].append(loss_potential)

      if (it % 500) == 0:
        pbar.set_postfix({"Loss": float(loss)})
      training_logs["cost_loss"].append(loss)
      
      if it % 5000 == 0 and it > 0 and callback is not None:
        callback(it, training_logs, self.transport)

      if it >= n_iters:
        break

    return training_logs

  def transport(
      self,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      n: int = 20,
      **kwargs: Any,
  ) -> jnp.ndarray:
    
    dt = 1.0 / n
    t_0 = 0.0
    loop_key = jax.random.PRNGKey(0)
    
    def solve_ode(state, x):
      def vector_field(t, y, args):
        dsdx_fn, key_s = args
        u = dsdx_fn(state.params, jnp.array(t)[None], y, y)
        #At_T = self.flow.compute_inverse_control_matrix(t_, x_).transpose()
        #U_t = self.flow.compute_potential(t, y)
        sigma = self.flow.compute_sigma_t(t)
        return -(u + sigma * jax.random.normal(key_s, shape=y.shape))

      dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)
      ode_term = diffrax.ODETerm(vector_field)
      saveat = diffrax.SaveAt(ts=jnp.linspace(0, 1, n))
      result = diffrax.diffeqsolve(
          ode_term,
          t0=0,
          t1=1,
          y0=x,
          args=(dsdx_fn, loop_key),
          solver=diffrax.Tsit5(),
          dt0=dt,
          saveat=saveat,
          **kwargs,
      )
      return None, result.ys
      # def move(carry, _):
      #   t_, x_, cost, key_ = carry
      #   u = dsdx_fn(state.params, t_ * jnp.ones([x.shape[0],1]), x_, x_)
      #   At_T = self.flow.compute_inverse_control_matrix(t_, x_).transpose()
      #   U_t = self.flow.compute_potential(t_, x_)
      #   sigma = self.flow.compute_sigma_t(t_)
      #   key_, key_s = jax.random.split(key_)
      #   x_ = x_ - dt * (u @ At_T + sigma * jax.random.normal(key_s, shape=x_.shape))
      #   t_ = t_ + dt
      #   cost += (0.5 * ((u @ At_T) * u).sum(-1).mean() * dt + self.potential_weight *U_t).mean() * dt
      #   return (t_, x_, cost, key_), x_
          
      # (_, _, cost, _), result = jax.lax.scan(move, (t_0, x, 0.0, loop_key), None, length=n)
      # return cost, result
    
    #cost, result = jax.jit(solve_ode)(self.state, x)
    cost, result = jax.jit(jax.vmap(solve_ode, in_axes=(None, 0), out_axes=1))(self.state, x)
    x_seq = [TimedX(t=t_0, x=x)]

    def compute_timesteps(carry, _):
      t_0, step = carry
      t_0 = t_0 + dt
      return (t_0, step+1), TimedX(t=t_0, x=result[step])
    
    x_seq = jax.lax.scan(compute_timesteps, init=(t_0, 0), length=n)[1]
    return cost, x_seq
