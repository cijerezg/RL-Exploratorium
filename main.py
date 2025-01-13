from envs.barkour_env import BarkourEnv, domain_randomize
from utils.helpers import create_video

from etils import epath

from datetime import datetime

import jax
from jax import numpy as jp
import numpy as np


import functools
import os

from flax.training import orbax_utils
from orbax import checkpoint as ocp

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model

os.environ['MUJOCO_GL'] = 'egl'
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

np.set_printoptions(precision=3, suppress=True, linewidth=100)


x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = 40, 0


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])
    print(x_data[-1])
    print(y_data[-1])
    print(ydataerr[-1])


envs.register_environment('barkour', BarkourEnv)

env_name = 'barkour'
env = envs.get_environment(env_name)

ckpt_path = epath.Path('/tmp/quadrupred_joystick/ckpts')
ckpt_path.mkdir(parents=True, exist_ok=True)

def policy_params_fn(current_step, make_policy, params):
    # save checkpoints
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f'{current_step}'
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)


make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))
train_fn = functools.partial(
      ppo.train, num_timesteps=100_000_000, num_evals=10,
      reward_scaling=1, episode_length=1000, normalize_observations=True,
      action_repeat=1, unroll_length=20, num_minibatches=32,
      num_updates_per_batch=4, discounting=0.97, learning_rate=3.0e-4,
      entropy_cost=1e-2, num_envs=2048, batch_size=256,
      network_factory=make_networks_factory,
      randomization_fn=domain_randomize,
      policy_params_fn=policy_params_fn,
      seed=0)


env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)


make_inference_fn, params, _= train_fn(environment=env,
                                       progress_fn=progress,
                                       eval_env=eval_env)



model_path = 'mjx_brax_quadruped_policy'
model.save_params(model_path, params)
params = model.load_params(model_path)


inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

eval_env = envs.get_environment(env_name)

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

x_vel = 1.0  #@param {type: "number"}
y_vel = 1.0  #@param {type: "number"}
ang_vel = 0.0  #@param {type: "number"}

the_command = jp.array([x_vel, y_vel, ang_vel])

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
state.info['command'] = the_command
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 500
render_every = 2

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)


frames = eval_env.render(rollout[::render_every], camera='track')
FPS = 1.0 / eval_env.dt / render_every


create_video(frames, 240, 320, FPS, 'videos', 'new_test')