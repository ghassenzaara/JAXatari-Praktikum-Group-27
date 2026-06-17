# C51 Reference (CleanRL) — Base for JAXtari Implementation

> Compiled reference for our C51 agent. Sources:
> - Docs: https://docs.cleanrl.dev/rl-algorithms/c51/
> - `c51_atari_jax.py`: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari_jax.py
> - `c51_jax.py`: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_jax.py
> - Original paper: Bellemare, Dabney & Munos (2017), *A Distributional Perspective on Reinforcement Learning*, ICML — https://arxiv.org/abs/1707.06887
>
> We base our agent on the two **JAX** variants only (Flax + Optax). `c51_atari_jax.py` → **pixel mode** (CNN); `c51_jax.py` → **object-centric mode** (MLP).

---

## 1. Overview

C51 gives DQN a *distributional* perspective: instead of predicting a single scalar Q-value per action, it predicts a **categorical distribution** over a fixed grid of return values (the "atoms"). With `n_atoms = 51` (hence "C51") spread over `[v_min, v_max]`, the network outputs a probability mass function (pmf) per action. The greedy action is the one whose distribution has the highest **expected value** `Σ p_i · z_i`.

Training minimizes the **cross-entropy** between the predicted pmf at step *t* and the **projected** Bellman-target pmf built from the target network at step *t+1*.

Everything else is standard DQN machinery: replay buffer, target network, ε-greedy exploration, Adam optimizer.

---

## 2. Which file maps to which mode

| Our mode | Base file | Network | Observation |
|---|---|---|---|
| **Pixel** (RGB / grayscale frames) | `c51_atari_jax.py` | CNN (3 conv + 2 dense) | stacked 84×84 frames |
| **Object-centric** | `c51_jax.py` | MLP (120 → 84) | flattened low-level feature vector |

The two files are **algorithmically identical** — same update, same projection, same loss. They differ only in (a) network architecture, (b) hyperparameters, (c) Atari preprocessing wrappers. This is the same `ppo_rgb` vs `ppo_oc` split our supervisor pointed at: *"it's just a change between conv layers to an MLP."*

---

## 3. Hyperparameters

### `c51_atari_jax.py` (pixel / CNN)

| Param | Value |
|---|---|
| `total_timesteps` | 10,000,000 |
| `learning_rate` | 2.5e-4 |
| `num_envs` | 1 (vectorized envs not supported in base) |
| `n_atoms` | 51 |
| `v_min` / `v_max` | -10 / 10 |
| `buffer_size` | 1,000,000 |
| `gamma` | 0.99 |
| `target_network_frequency` | 10,000 |
| `batch_size` | 32 |
| `start_e` / `end_e` | 1 / 0.01 |
| `exploration_fraction` | 0.10 |
| `learning_starts` | 80,000 |
| `train_frequency` | 4 |
| optimizer | `optax.adam(lr, eps=0.01/batch_size)` |

### `c51_jax.py` (classic control / MLP)

| Param | Value |
|---|---|
| `total_timesteps` | 500,000 |
| `learning_rate` | 2.5e-4 |
| `num_envs` | 1 |
| `n_atoms` | **101** |
| `v_min` / `v_max` | **-100 / 100** |
| `buffer_size` | 10,000 |
| `gamma` | 0.99 |
| `target_network_frequency` | 500 |
| `batch_size` | 128 |
| `start_e` / `end_e` | 1 / 0.05 |
| `exploration_fraction` | 0.5 |
| `learning_starts` | 10,000 |
| `train_frequency` | 10 |
| optimizer | `optax.adam(lr, eps=0.01/batch_size)` |

> Note the OC/MLP config uses **more atoms** (101) and a **wider value range** (±100) because CartPole-style returns are larger and differently scaled than clipped Atari rewards (±10). When we tune OC mode on JAXtari, `n_atoms`, `v_min`, `v_max` must be set to the actual return range of each game.

---

## 4. Network architectures

### Pixel — CNN (`c51_atari_jax.py`)

```python
class QNetwork(nn.Module):
    action_dim: int
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))   # NCHW -> NHWC
        x = x / 255.0
        x = nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x); x = nn.relu(x)
        x = nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x); x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x); x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x); x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        x = nn.softmax(x, axis=-1)           # pmfs
        return x
```

### Object-centric — MLP (`c51_jax.py`)

```python
class QNetwork(nn.Module):
    action_dim: int
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(120)(x); x = nn.relu(x)
        x = nn.Dense(84)(x);  x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        x = nn.softmax(x, axis=-1)           # pmfs
        return x
```

Both output shape `(batch, action_dim, n_atoms)` after a softmax over the atom axis → a valid pmf per action. **The conv stack is the only structural difference.**

### Custom TrainState

Both carry the target params and the atom grid on the train state:

```python
class TrainState(TrainState):
    target_params: flax.core.FrozenDict
    atoms: jnp.ndarray
```

Atoms are built once with numpy to avoid `jnp.linspace` numerical error:
```python
atoms = jnp.asarray(np.linspace(v_min, v_max, num=n_atoms))
```

---

## 5. Core algorithm (identical in both files)

### Greedy action
```python
pmfs   = q_network.apply(params, obs)          # (B, A, n_atoms)
q_vals = (pmfs * atoms).sum(axis=-1)           # expected value per action (B, A)
action = q_vals.argmax(axis=-1)
```

### Distributional Bellman target + projection
```python
next_pmfs = q_network.apply(target_params, next_obs)     # (B, A, n_atoms)
next_vals = (next_pmfs * atoms).sum(-1)
next_action = jnp.argmax(next_vals, -1)
next_pmfs = next_pmfs[arange(B), next_action]            # greedy pmf (B, n_atoms)

next_atoms = rewards + gamma * atoms * (1 - dones)       # shifted support
delta_z = atoms[1] - atoms[0]
tz = jnp.clip(next_atoms, v_min, v_max)
b  = (tz - v_min) / delta_z
l  = clip(floor(b), 0, n_atoms-1)
u  = clip(ceil(b),  0, n_atoms-1)
# (l == u) term handles b landing exactly on an integer bin
d_m_l = (u + (l == u) - b) * next_pmfs
d_m_u = (b - l) * next_pmfs
# scatter-add mass into target_pmfs via lax.fori_loop over batch
```

### Loss (cross-entropy)
```python
pmfs     = q_network.apply(q_params, obs)
old_pmfs = pmfs[arange(B), actions]                      # pmf of taken action
old_pmfs = jnp.clip(old_pmfs, 1e-5, 1 - 1e-5)
loss = (-(target_pmfs * jnp.log(old_pmfs)).sum(-1)).mean()
```

`update` is wrapped with `@jax.jit`; gradients via `jax.value_and_grad(loss, has_aux=True)`; target net synced with `optax.incremental_update(params, target_params, 1)` (a hard copy) every `target_network_frequency` steps.

---

## 6. Logged metrics

- `charts/episodic_return` — episodic return of the game
- `charts/episodic_length` — episode length
- `charts/SPS` — steps per second
- `losses/loss` — cross-entropy between the *t* state-value distribution and the projected *t+1* distribution
- `losses/q_values` — `(old_pmfs * atoms).sum(1)` averaged over the batch; sum of P(return = x)·x. Useful for spotting over/under-estimation.

---

## 7. Implementation details (from docs)

- Based on Bellemare et al. (2017) with two deviations:
  1. The paper injects stochasticity by rejecting the agent's action with `p = 0.25` per frame; CleanRL does **not**.
  2. CleanRL reports the **training** episodic returns directly (self-contained eval), whereas the paper used a separate evaluation process (`--end-e=0.001`).
- `c51_atari_jax.py` is ~25% faster than the PyTorch `c51_atari.py`; `c51_jax.py` is ~55% faster than `c51.py`.
- Atari preprocessing wrappers (in `make_env`): `NoopResetEnv(30)`, `MaxAndSkipEnv(4)`, `EpisodicLifeEnv`, `FireResetEnv` (if applicable), `ClipRewardEnv`, `ResizeObservation(84,84)`, `GrayScaleObservation`, `FrameStack(4)`.

---

## 8. Benchmark results (reference targets)

### Atari (10M steps, `c51_atari_jax.py`)

| Environment | `c51_atari_jax.py` (10M) | `c51_atari.py` (10M) | Bellemare 2017 (50M) | Rainbow / Hessel 2018 |
|---|---|---|---|---|
| Breakout | 448.56 ± 17.02 | 461.86 ± 69.65 | 748 | ~500 @10M, ~600 @50M |
| Pong | 19.88 ± 0.31 | 19.46 ± 0.70 | 20.9 | ~20 |
| BeamRider | 9504.91 ± 709.69 | 9592.90 ± 2270.15 | 14,074 | ~12000 @10M |

### Classic control (`c51_jax.py`)

| Environment | `c51_jax.py` | `c51.py` |
|---|---|---|
| CartPole-v1 | 491.07 ± 9.70 | 481.20 ± 20.53 |
| Acrobot-v1 | -86.74 ± 2.19 | -87.70 ± 5.52 |
| MountainCar-v0 | -174.30 ± 36.35 | -166.38 ± 27.94 |

(For our report, the Pong/BeamRider Atari numbers are the relevant baselines, and the original-paper column is the "vs original" comparison.)

---

## 9. What we must change for JAXtari (porting notes)

These are the deltas between the CleanRL baseline and our project (see project `CLAUDE.md`):

1. **Replay buffer.** CleanRL's `ReplayBuffer` (from `cleanrl_utils.buffers`) requires gymnasium `Space` objects, which JAXAtari doesn't expose. Per supervisor: **extend JAXAtari's spaces** — add a new class inheriting JAXAtari's base `Space`, implemented in JAX, matching the existing jaxtari spaces. Do **not** bolt on a separate numpy ring buffer.
2. **Both observation modes in one agent.** Support pixel (CNN) and object-centric (MLP) via the wrappers (`PixelObsWrapper` vs `ObjectCentricWrapper` + `FlattenObservationWrapper`). Architecture swap = conv stack ↔ MLP; keep everything else as close to baseline as possible.
3. **Separate HP sets** for pixel vs OC (the two tables above are the starting points). For OC mode, set `n_atoms` / `v_min` / `v_max` to the actual return range of each JAXtari game.
4. **Optimizer config switch.** Expose `optimizer: "adam" | "rmsprop"` (default `adam`, matching CleanRL) so the original-paper RMSProp is reproducible.
5. **`num_envs`.** Base config = 8 parallel envs (project default), HPs as close to baseline as possible. C51 is DQN-family → doesn't scale favorably with `n_envs` due to the replay buffer; scale up for the "optimized" config but expect quality degradation. If no time improvement, no separate optimized config needed.
6. **Vectorized envs.** The CleanRL JAX C51 base asserts `num_envs == 1`. We need true vectorized rollout for JAXtari's 8-env default — this is one of the real porting tasks.
7. **`lax.scan`.** Use `jax.lax.scan` for rollout loops wherever applicable (project-wide convention), rather than Python for-loop + jitted update.

---

## 10. Full source — `c51_atari_jax.py`

```python
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51_atari_jaxpy
import os
import random
import time
from dataclasses import dataclass

# see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    total_timesteps: int = 10000000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    n_atoms: int = 51
    v_min: float = -10
    v_max: float = 10
    buffer_size: int = 1000000
    gamma: float = 0.99
    target_network_frequency: int = 10000
    batch_size: int = 32
    start_e: float = 1
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80000
    train_frequency: int = 4


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    action_dim: int
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        x = nn.softmax(x, axis=-1)  # pmfs
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict
    atoms: jnp.ndarray


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs, _ = envs.reset(seed=args.seed)

    q_network = QNetwork(action_dim=envs.single_action_space.n, n_atoms=args.n_atoms)

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        # directly using jnp.linspace leads to numerical errors
        atoms=jnp.asarray(np.linspace(args.v_min, args.v_max, num=args.n_atoms)),
        tx=optax.adam(learning_rate=args.learning_rate, eps=0.01 / args.batch_size),
    )

    q_network.apply = jax.jit(q_network.apply)
    # This step is not necessary as init called on same observation and key will always lead to same initializations
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        next_pmfs = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions, num_atoms)
        next_vals = (next_pmfs * q_state.atoms).sum(axis=-1)  # (batch_size, num_actions)
        next_action = jnp.argmax(next_vals, axis=-1)  # (batch_size,)
        next_pmfs = next_pmfs[np.arange(next_pmfs.shape[0]), next_action]
        next_atoms = rewards + args.gamma * q_state.atoms * (1 - dones)
        # projection
        delta_z = q_state.atoms[1] - q_state.atoms[0]
        tz = jnp.clip(next_atoms, a_min=(args.v_min), a_max=(args.v_max))

        b = (tz - args.v_min) / delta_z
        l = jnp.clip(jnp.floor(b), a_min=0, a_max=args.n_atoms - 1)
        u = jnp.clip(jnp.ceil(b), a_min=0, a_max=args.n_atoms - 1)
        # (l == u).astype(jnp.float) handles the case where bj is exactly an integer
        # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
        d_m_l = (u + (l == u).astype(jnp.float32) - b) * next_pmfs
        d_m_u = (b - l) * next_pmfs
        target_pmfs = jnp.zeros_like(next_pmfs)

        def project_to_bins(i, val):
            val = val.at[i, l[i].astype(jnp.int32)].add(d_m_l[i])
            val = val.at[i, u[i].astype(jnp.int32)].add(d_m_u[i])
            return val

        target_pmfs = jax.lax.fori_loop(0, target_pmfs.shape[0], project_to_bins, target_pmfs)

        def loss(q_params, observations, actions, target_pmfs):
            pmfs = q_network.apply(q_params, observations)
            old_pmfs = pmfs[np.arange(pmfs.shape[0]), actions.squeeze()]

            old_pmfs_l = jnp.clip(old_pmfs, a_min=1e-5, a_max=1 - 1e-5)
            loss = (-(target_pmfs * jnp.log(old_pmfs_l)).sum(-1)).mean()
            return loss, (old_pmfs * q_state.atoms).sum(-1)

        (loss_value, old_values), grads = jax.value_and_grad(loss, has_aux=True)(
            q_state.params, observations, actions, target_pmfs
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, old_values, q_state

    @jax.jit
    def get_action(q_state, obs):
        pmfs = q_network.apply(q_state.params, obs)
        q_vals = (pmfs * q_state.atoms).sum(axis=-1)
        actions = q_vals.argmax(axis=-1)
        return actions

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = get_action(q_state, obs)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            loss, old_val, q_state = update(
                q_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.numpy(),
                data.dones.numpy(),
            )

            if global_step % 100 == 0:
                writer.add_scalar("losses/loss", jax.device_get(loss), global_step)
                writer.add_scalar("losses/q_values", jax.device_get(old_val.mean()), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_state.params,
            "args": vars(args),
        }
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(model_data))
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.c51_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "C51", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
```

---

## 11. Full source — `c51_jax.py`

```python
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51_jaxpy
import os
import random
import time
from dataclasses import dataclass

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    n_atoms: int = 101
    v_min: float = -100
    v_max: float = 100
    buffer_size: int = 10000
    gamma: float = 0.99
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    action_dim: int
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        x = nn.softmax(x, axis=-1)  # pmfs
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict
    atoms: jnp.ndarray


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs, _ = envs.reset(seed=args.seed)
    q_network = QNetwork(action_dim=envs.single_action_space.n, n_atoms=args.n_atoms)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        # directly using jnp.linspace leads to numerical errors
        atoms=jnp.asarray(np.linspace(args.v_min, args.v_max, num=args.n_atoms)),
        tx=optax.adam(learning_rate=args.learning_rate, eps=0.01 / args.batch_size),
    )
    q_network.apply = jax.jit(q_network.apply)
    # This step is not necessary as init called on same observation and key will always lead to same initializations
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        next_pmfs = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions, num_atoms)
        next_vals = (next_pmfs * q_state.atoms).sum(axis=-1)  # (batch_size, num_actions)
        next_action = jnp.argmax(next_vals, axis=-1)  # (batch_size,)
        next_pmfs = next_pmfs[np.arange(next_pmfs.shape[0]), next_action]
        next_atoms = rewards + args.gamma * q_state.atoms * (1 - dones)
        # projection
        delta_z = q_state.atoms[1] - q_state.atoms[0]
        tz = jnp.clip(next_atoms, a_min=(args.v_min), a_max=(args.v_max))

        b = (tz - args.v_min) / delta_z
        l = jnp.clip(jnp.floor(b), a_min=0, a_max=args.n_atoms - 1)
        u = jnp.clip(jnp.ceil(b), a_min=0, a_max=args.n_atoms - 1)
        # (l == u).astype(jnp.float) handles the case where bj is exactly an integer
        # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
        d_m_l = (u + (l == u).astype(jnp.float32) - b) * next_pmfs
        d_m_u = (b - l) * next_pmfs
        target_pmfs = jnp.zeros_like(next_pmfs)

        def project_to_bins(i, val):
            val = val.at[i, l[i].astype(jnp.int32)].add(d_m_l[i])
            val = val.at[i, u[i].astype(jnp.int32)].add(d_m_u[i])
            return val

        target_pmfs = jax.lax.fori_loop(0, target_pmfs.shape[0], project_to_bins, target_pmfs)

        def loss(q_params, observations, actions, target_pmfs):
            pmfs = q_network.apply(q_params, observations)
            old_pmfs = pmfs[np.arange(pmfs.shape[0]), actions.squeeze()]

            old_pmfs_l = jnp.clip(old_pmfs, a_min=1e-5, a_max=1 - 1e-5)
            loss = (-(target_pmfs * jnp.log(old_pmfs_l)).sum(-1)).mean()
            return loss, (old_pmfs * q_state.atoms).sum(-1)

        (loss_value, old_values), grads = jax.value_and_grad(loss, has_aux=True)(
            q_state.params, observations, actions, target_pmfs
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, old_values, q_state

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            pmfs = q_network.apply(q_state.params, obs)
            q_vals = (pmfs * q_state.atoms).sum(axis=-1)
            actions = q_vals.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            loss, old_val, q_state = update(
                q_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.numpy(),
                data.dones.numpy(),
            )

            if global_step % 100 == 0:
                writer.add_scalar("losses/loss", jax.device_get(loss), global_step)
                writer.add_scalar("losses/q_values", jax.device_get(old_val.mean()), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_state.params,
            "args": vars(args),
        }
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(model_data))
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.c51_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "C51", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
```
