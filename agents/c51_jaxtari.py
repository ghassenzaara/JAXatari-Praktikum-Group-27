"""
Single-file C51 (Categorical DQN) for JAXAtari — pixel and object-centric modes.

Topic 26, TU Darmstadt JAXtari praktikum.

Credits / sources
-----------------
- Algorithm: Bellemare, Dabney & Munos (2017), "A Distributional Perspective on
  Reinforcement Learning", ICML. https://arxiv.org/abs/1707.06887
- Reference implementations (CleanRL, Costa Huang et al., MIT-licensed):
    * cleanrl/c51_atari_jax.py  -> CNN / pixel agent + full training scaffolding
    * cleanrl/c51_jax.py        -> MLP head for low-level / object-centric features
  https://github.com/vwxyzjn/cleanrl  (docs: https://docs.cleanrl.dev/rl-algorithms/c51/)
- The distributional projection + cross-entropy loss are ported verbatim (math-wise)
  from the CleanRL JAX C51 `update` function. See `C51 doccumentation.md` in the repo
  root for the annotated reference this file was built from.

What is different from CleanRL (and why)
----------------------------------------
1. Environment: JAXAtari (GPU, JAX-native) instead of gymnasium ALE. We use the
   JAXAtari wrappers (AtariWrapper + Pixel/ObjectCentric + Log) directly.
2. Replay buffer: CleanRL's ReplayBuffer needs gymnasium Spaces (which JAXAtari does
   not expose). We use a small self-contained, jittable JAX replay buffer whose
   storage is sized from the wrapper's JAXAtari `Box` observation space
   (`env.observation_space().shape/.dtype`). No numpy ring buffer, no gymnasium Space.
3. Both observation modes in one file (CNN for pixels, MLP for object-centric),
   selected by `--obs-mode`. Architecture is the only structural difference
   (cf. ppo_rgb vs ppo_oc) — hyperparameters are otherwise shared, since both modes
   run the same games with the same clipped (+/-1) reward scale.
4. Vectorized: `--num-envs` (default 8) parallel envs via `jax.vmap`.
5. The whole rollout + training loop runs inside `jax.lax.scan` (project convention),
   not a Python for-loop with a per-step jitted update.

NOTE: This is a first full port. It is intended to be run/debugged on the GPU box in
WSL (the dev sandbox has no GPU and cannot reach the WSL filesystem). Assumptions that
should be verified on first run are flagged with `# VERIFY`.
"""

import os
import math
import time
from dataclasses import dataclass
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax import struct
from flax.training.train_state import TrainState

# RTPT (Remaining Time to Process Title): required by the TU ML student-pool rules so
# lab admins can see who is running what and for how long. Optional import so the agent
# still runs on machines without it (e.g. local dev). Enable/disable via --rtpt-initials.
try:
    from rtpt import RTPT
except ImportError:
    RTPT = None

import jaxatari
from jaxatari.wrappers import (
    AtariWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    FlattenObservationWrapper,
    NormalizeObservationWrapper,
    LogWrapper,
)

# Results layout: results/<ALGO_NAME>/<Game>/<ModeFolder>/<files>, e.g.
# results/C51/Pong/Pixel/ and results/C51/Pong/ObjectCentric/. Sibling folders CleanRL/
# (reference CSVs) and Report/ (.tex/.pdf + figures) are managed by the comparison tooling.
# The game arg is lowercase (the JAXAtari key, e.g. "pong"); folders are TitleCase.
# GAME_DISPLAY maps multi-word games that .capitalize() would mangle ("mspacman"->"MsPacman").
ALGO_NAME = "C51"
GAME_DISPLAY = {
    "mspacman": "MsPacman",
    "montezumarevenge": "MontezumaRevenge",
    "beamrider": "BeamRider",
}
MODE_DISPLAY = {"pixel": "Pixel", "object_centric": "ObjectCentric"}


def game_dir_name(game: str) -> str:
    """TitleCase folder name for a JAXAtari game key (results/<algo>/<Game>/)."""
    return GAME_DISPLAY.get(game, game.capitalize())


def mode_dir_name(obs_mode: str) -> str:
    """Run-mode subfolder name (results/<algo>/<Game>/<Pixel|ObjectCentric>/)."""
    return MODE_DISPLAY.get(obs_mode, obs_mode)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class Args:
    exp_name: str = "c51_jaxtari"
    seed: int = 1
    track: bool = False
    """if toggled, log to Weights and Biases"""
    wandb_project_name: str = "jaxtari-baselines"
    wandb_entity: str = None

    # Environment
    game: str = "pong"
    """JAXAtari game name (e.g. pong, seaquest, breakout)"""
    obs_mode: str = "pixel"
    """'pixel' (CNN) or 'object_centric' (MLP)"""
    num_envs: int = 8
    """number of parallel environments (vmapped)"""
    sticky_actions: float = 0.0
    """sticky-action probability; 0.0 matches the CleanRL NoFrameskip baseline"""

    # C51 / DQN hyperparameters (shared across modes — only the network differs)
    total_timesteps: int = 10_000_000
    """total environment frames summed across all envs"""
    learning_rate: float = 2.5e-4
    optimizer: str = "adam"
    """'adam' (CleanRL baseline) or 'rmsprop' (original DQN paper)"""
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    buffer_size: int | None = None
    """replay capacity. The buffer lives on-device (GPU) because it sits inside the
    lax.scan carry. Pixel frames (4x84x84 uint8) are large (~55 KB/transition), so
    capacity is VRAM-bound; OC vectors are tiny (~1 KB/transition) so the full CleanRL
    buffer fits. If left unset, defaults per mode: 50_000 (pixel) / 1_000_000 (OC) —
    sized for an 11 GB 2080 Ti. Lower the pixel value on <8GB GPUs (e.g. 10_000)."""
    gamma: float = 0.99
    target_network_frequency: int = 10_000
    """frames between hard target-network syncs"""
    batch_size: int = 32
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80_000
    """frames before learning begins"""
    train_frequency: int = 4
    """frames between gradient steps (per-env-aware; see training loop)"""

    # Logging / output
    num_logs: int = 100
    """how many times to break out of scan to log + checkpoint metrics"""
    save_results: bool = True
    results_dir: str = "results"
    run_tag: str = ""
    """optional suffix on output filenames (e.g. "512x512", "seed2") so variant runs of the
    same (game, mode) don't overwrite each other. Empty = no suffix (default file names)."""
    rtpt_initials: str = "GZ"
    """name initials shown in the RTPT process title (lab rule). Set to '' to disable RTPT."""


# --------------------------------------------------------------------------------------
# Networks  (output: per-action categorical pmf over `n_atoms` return atoms)
# --------------------------------------------------------------------------------------
class CNNQNetwork(nn.Module):
    """Pixel mode. Expects stacked frames (B, stack, H, W, C); uses stack*C as channels."""
    action_dim: int
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        # (B, stack, H, W, C) -> (B, H, W, stack*C)
        b = x.shape[0]
        x = x.astype(jnp.float32) / 255.0
        x = jnp.transpose(x, (0, 2, 3, 1, 4))         # (B, H, W, stack, C)
        x = x.reshape((b, x.shape[1], x.shape[2], -1))  # (B, H, W, stack*C)
        x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x))
        x = x.reshape((b, -1))
        x = nn.relu(nn.Dense(512)(x))
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((b, self.action_dim, self.n_atoms))
        return nn.softmax(x, axis=-1)


class MLPQNetwork(nn.Module):
    """Object-centric mode. Expects a flat feature vector (B, features).

    Hidden sizes: tapered (64, 32, 16), per tutor feedback (2026-07-07) that the
    earlier (512, 512) net was too wide. Once inputs are normalized to [0,1] by
    NormalizeObservationWrapper (see make_env), a small net suffices — the real fix
    for the original OC collapse was normalization, not width. Width is exposed as
    `hidden_sizes` so it can be re-tuned per game without editing the class.
    (Earlier notes: JAXAtari's ppo_jaxatari_scan.py OC baseline uses Dense(461)->
    Dense(512); CleanRL's CartPole c51_jax.py uses (120, 84).) No /255 here — inputs
    are already scaled.
    """
    action_dim: int
    n_atoms: int
    hidden_sizes: tuple = (64, 32, 16)

    @nn.compact
    def __call__(self, x):
        b = x.shape[0]
        x = x.astype(jnp.float32)
        for h in self.hidden_sizes:
            x = nn.relu(nn.Dense(h)(x))
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((b, self.action_dim, self.n_atoms))
        return nn.softmax(x, axis=-1)


class C51TrainState(TrainState):
    target_params: flax.core.FrozenDict
    atoms: jnp.ndarray


# --------------------------------------------------------------------------------------
# JAX-native replay buffer (jittable, lives in the scan carry)
# --------------------------------------------------------------------------------------
@struct.dataclass
class ReplayBuffer:
    obs: jnp.ndarray        # (capacity, *obs_shape)
    next_obs: jnp.ndarray   # (capacity, *obs_shape)
    actions: jnp.ndarray    # (capacity,)
    rewards: jnp.ndarray    # (capacity,)
    dones: jnp.ndarray      # (capacity,)  -- terminal flag used for bootstrap masking
    pos: jnp.ndarray        # scalar int32, next write index
    size: jnp.ndarray       # scalar int32, current number of stored transitions


def buffer_init(capacity: int, obs_shape, obs_dtype) -> ReplayBuffer:
    return ReplayBuffer(
        obs=jnp.zeros((capacity, *obs_shape), dtype=obs_dtype),
        next_obs=jnp.zeros((capacity, *obs_shape), dtype=obs_dtype),
        actions=jnp.zeros((capacity,), dtype=jnp.int32),
        rewards=jnp.zeros((capacity,), dtype=jnp.float32),
        dones=jnp.zeros((capacity,), dtype=jnp.float32),
        pos=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
    )


def buffer_add(buf: ReplayBuffer, obs, actions, rewards, next_obs, dones, capacity: int) -> ReplayBuffer:
    """Insert a batch of `num_envs` transitions (ring-buffer, wraps around)."""
    n = obs.shape[0]
    idx = (buf.pos + jnp.arange(n)) % capacity
    return buf.replace(
        obs=buf.obs.at[idx].set(obs),
        next_obs=buf.next_obs.at[idx].set(next_obs),
        actions=buf.actions.at[idx].set(actions.astype(jnp.int32)),
        rewards=buf.rewards.at[idx].set(rewards.astype(jnp.float32)),
        dones=buf.dones.at[idx].set(dones.astype(jnp.float32)),
        pos=(buf.pos + n) % capacity,
        size=jnp.minimum(buf.size + n, capacity),
    )


def buffer_sample(buf: ReplayBuffer, key, batch_size: int):
    """Uniform random sample of `batch_size` transitions from the filled region."""
    idx = jax.random.randint(key, (batch_size,), 0, buf.size)
    return (buf.obs[idx], buf.actions[idx], buf.rewards[idx], buf.next_obs[idx], buf.dones[idx])


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def linear_schedule(start_e: float, end_e: float, duration: float, t):
    slope = (end_e - start_e) / duration
    return jnp.maximum(slope * t + start_e, end_e)


def make_env(args: Args):
    """Build the wrapped JAXAtari env for the chosen observation mode."""
    base = jaxatari.make(args.game)
    atari = AtariWrapper(
        base,
        sticky_actions=args.sticky_actions,
        episodic_life=True,
        first_fire=True,
        noop_max=30,
    )
    if args.obs_mode == "pixel":
        env = PixelObsWrapper(
            atari,
            do_pixel_resize=True,
            pixel_resize_shape=(84, 84),
            grayscale=True,
            frame_stack_size=4,
            frame_skip=4,
            max_pooling=True,
            clip_reward=True,
        )
    elif args.obs_mode == "object_centric":
        # NormalizeObservationWrapper is ESSENTIAL here: ObjectCentricWrapper emits raw
        # pixel-space coordinates (x in [0,160], y in [0,210], widths, ...). The MLP has
        # no input scaling of its own (unlike the CNN's /255), so without this the raw
        # large-magnitude features saturate the tiny net, the per-atom softmax collapses
        # to ~one-hot, and the greedy policy locks to a single bad action (Pong -> -21).
        # NormalizeObservationWrapper scales each feature to [0,1] using the space bounds.
        # Ordering matches JAXAtari's own ppo_oc baseline: Flatten(Normalize(OC(...))).
        env = FlattenObservationWrapper(
            NormalizeObservationWrapper(
                ObjectCentricWrapper(atari, frame_stack_size=4, frame_skip=4, clip_reward=True)
            )
        )
    else:
        raise ValueError(f"obs_mode must be 'pixel' or 'object_centric', got {args.obs_mode}")
    return LogWrapper(env)


# --------------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------------
def main(args: Args):
    run_name = f"{args.game}__{args.exp_name}__{args.obs_mode}__{args.seed}__{int(time.time())}"
    print(f"JAX devices: {jax.devices()}")
    print(f"Run: {run_name}")

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   name=run_name, config=vars(args), save_code=True)

    key = jax.random.PRNGKey(args.seed)
    key, net_key, env_key = jax.random.split(key, 3)

    # ---- env + spaces -------------------------------------------------------------
    env = make_env(args)
    n_actions = int(env.action_space().n)                       # VERIFY proxies to base env
    obs_space = env.observation_space()                         # JAXAtari Box (stacked)
    obs_shape = tuple(obs_space.shape)
    obs_dtype = obs_space.dtype
    print(f"obs_mode={args.obs_mode}  obs_shape={obs_shape}  dtype={obs_dtype}  n_actions={n_actions}")

    # Mode-aware default replay capacity (on-device buffer is VRAM-bound for pixels).
    # Pixel frames are ~55 KB/transition (obs+next_obs), so on an 11 GB 2080 Ti ~50k
    # fits comfortably (~2.8 GB) while the CleanRL 1M (~56 GB) is impossible. OC vectors
    # are ~1 KB/transition, so the full CleanRL 1M buffer costs only ~1 GB — no VRAM issue.
    if args.buffer_size is None:
        args.buffer_size = 50_000 if args.obs_mode == "pixel" else 1_000_000
    print(f"buffer_size={args.buffer_size}")

    reset_fn = jax.vmap(env.reset)
    step_fn = jax.vmap(env.step, in_axes=(0, 0))

    env_keys = jax.random.split(env_key, args.num_envs)
    obs, env_state = reset_fn(env_keys)                         # obs: (num_envs, *obs_shape)

    # ---- network ------------------------------------------------------------------
    Net = CNNQNetwork if args.obs_mode == "pixel" else MLPQNetwork
    q_network = Net(action_dim=n_actions, n_atoms=args.n_atoms)
    sample_obs = jnp.zeros((1, *obs_shape), dtype=obs_dtype)

    if args.optimizer == "adam":
        tx = optax.adam(learning_rate=args.learning_rate, eps=0.01 / args.batch_size)
    elif args.optimizer == "rmsprop":
        tx = optax.rmsprop(learning_rate=args.learning_rate, eps=0.01 / args.batch_size,
                           decay=0.95, initial_scale=0.0)
    else:
        raise ValueError(f"optimizer must be 'adam' or 'rmsprop', got {args.optimizer}")

    params = q_network.init(net_key, sample_obs)
    q_state = C51TrainState.create(
        apply_fn=q_network.apply,
        params=params,
        target_params=params,
        atoms=jnp.asarray(np.linspace(args.v_min, args.v_max, num=args.n_atoms), dtype=jnp.float32),
        tx=tx,
    )

    buffer = buffer_init(args.buffer_size, obs_shape, obs_dtype)

    # ---- C51 core: greedy action + distributional Bellman update ------------------
    def get_action(q_state, obs):
        pmfs = q_network.apply(q_state.params, obs)             # (B, A, n_atoms)
        q_vals = (pmfs * q_state.atoms).sum(axis=-1)            # (B, A)
        return jnp.argmax(q_vals, axis=-1)                      # (B,)

    def c51_update(q_state, observations, actions, rewards, next_observations, dones):
        next_pmfs = q_network.apply(q_state.target_params, next_observations)   # (B, A, n_atoms)
        next_vals = (next_pmfs * q_state.atoms).sum(axis=-1)
        next_action = jnp.argmax(next_vals, axis=-1)
        b_idx = jnp.arange(next_pmfs.shape[0])
        next_pmfs = next_pmfs[b_idx, next_action]                               # (B, n_atoms)

        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        next_atoms = rewards + args.gamma * q_state.atoms * (1.0 - dones)       # (B, n_atoms)

        # projection onto the fixed atom support
        delta_z = q_state.atoms[1] - q_state.atoms[0]
        tz = jnp.clip(next_atoms, args.v_min, args.v_max)
        b = (tz - args.v_min) / delta_z
        l = jnp.clip(jnp.floor(b), 0, args.n_atoms - 1)
        u = jnp.clip(jnp.ceil(b), 0, args.n_atoms - 1)
        d_m_l = (u + (l == u).astype(jnp.float32) - b) * next_pmfs
        d_m_u = (b - l) * next_pmfs
        target_pmfs = jnp.zeros_like(next_pmfs)

        def project_to_bins(i, val):
            val = val.at[i, l[i].astype(jnp.int32)].add(d_m_l[i])
            val = val.at[i, u[i].astype(jnp.int32)].add(d_m_u[i])
            return val

        target_pmfs = jax.lax.fori_loop(0, target_pmfs.shape[0], project_to_bins, target_pmfs)

        def loss_fn(params):
            pmfs = q_network.apply(params, observations)
            old_pmfs = pmfs[b_idx, actions.squeeze()]
            old_pmfs_l = jnp.clip(old_pmfs, 1e-5, 1 - 1e-5)
            loss = (-(target_pmfs * jnp.log(old_pmfs_l)).sum(-1)).mean()
            return loss, (old_pmfs * q_state.atoms).sum(-1).mean()

        (loss_value, q_val), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return q_state, loss_value, q_val

    # ---- one environment iteration (acts on all envs once) ------------------------
    exploration_frames = args.exploration_fraction * args.total_timesteps

    def step(carry, _):
        q_state, buffer, obs, env_state, key, global_step = carry
        key, act_key, expl_key, sample_key = jax.random.split(key, 4)

        epsilon = linear_schedule(args.start_e, args.end_e, exploration_frames, global_step)
        greedy = get_action(q_state, obs)
        rand = jax.random.randint(act_key, (args.num_envs,), 0, n_actions)
        explore = jax.random.uniform(expl_key, (args.num_envs,)) < epsilon
        actions = jnp.where(explore, rand, greedy)

        next_obs, env_state, reward, terminated, truncated, info = step_fn(env_state, actions)
        buffer = buffer_add(buffer, obs, actions, reward, next_obs,
                            terminated.astype(jnp.float32), args.buffer_size)
        global_step = global_step + args.num_envs

        # gradient step. train_frequency is in frames; we fire an update whenever the
        # frame counter crosses a train_frequency boundary this iteration. When
        # num_envs >= train_frequency that is every iteration; when smaller it spaces
        # updates out to match the baseline cadence.
        crossed = (global_step // args.train_frequency) != ((global_step - args.num_envs) // args.train_frequency)
        do_train = jnp.logical_and(
            jnp.logical_and(global_step > args.learning_starts, buffer.size >= args.batch_size),
            crossed,
        )

        def _train(qs):
            return c51_update(qs, *buffer_sample(buffer, sample_key, args.batch_size))

        def _skip(qs):
            return qs, jnp.float32(0.0), jnp.float32(0.0)

        q_state, loss, q_val = jax.lax.cond(do_train, _train, _skip, q_state)

        # hard target sync at frequency boundaries
        do_sync = (global_step // args.target_network_frequency) != \
                  ((global_step - args.num_envs) // args.target_network_frequency)
        q_state = jax.lax.cond(
            do_sync,
            lambda qs: qs.replace(target_params=optax.incremental_update(qs.params, qs.target_params, 1.0)),
            lambda qs: qs,
            q_state,
        )

        metrics = {
            "returns": info["returned_episode_returns"],   # (num_envs,)
            "lengths": info["returned_episode_lengths"],   # (num_envs,)
            "finished": info["returned_episode"],          # (num_envs,) bool
            "loss": loss,
            "q_val": q_val,
            "trained": do_train.astype(jnp.float32),       # 1.0 on gradient-step iters
            "epsilon": epsilon,
        }
        return (q_state, buffer, next_obs, env_state, key, global_step), metrics

    @partial(jax.jit, static_argnames=("n_iters",))
    def run_chunk(carry, n_iters):
        return jax.lax.scan(step, carry, None, length=n_iters)

    # ---- driver: chunked scan for periodic logging --------------------------------
    total_iters = args.total_timesteps // args.num_envs
    chunk_iters = max(1, total_iters // args.num_logs)
    n_chunks = math.ceil(total_iters / chunk_iters)
    carry = (q_state, buffer, obs, env_state, key, jnp.array(0, dtype=jnp.int32))

    # RTPT: one .step() per logging chunk, so max_iterations == number of chunks.
    rtpt = None
    if RTPT is not None and args.rtpt_initials:
        rtpt = RTPT(name_initials=args.rtpt_initials,
                    experiment_name=f"C51-{args.game}-{args.obs_mode}",
                    max_iterations=n_chunks)
        rtpt.start()
    elif RTPT is None and args.rtpt_initials:
        print("(RTPT not installed — process title won't show remaining time. "
              "`pip install rtpt` on the lab machine.)")

    # Per-chunk records, one row per log point. Columns mirror CleanRL's TensorBoard panels
    # (charts/episodic_return, charts/episodic_length, losses/loss, losses/q_values,
    # charts/epsilon, charts/SPS) so the report can show the same dashboard offline.
    history = []  # list of dicts (see COLUMNS below)
    COLUMNS = ["global_step", "episodic_return", "episodic_length",
               "loss", "q_value", "epsilon", "sps"]
    start_time = time.time()
    prev_step = 0
    prev_time = start_time
    done_iters = 0
    while done_iters < total_iters:
        n = min(chunk_iters, total_iters - done_iters)
        carry, metrics = run_chunk(carry, n_iters=n)
        done_iters += n
        global_step = int(carry[5])
        now = time.time()

        finished = np.asarray(metrics["finished"])
        returns = np.asarray(metrics["returns"])
        lengths = np.asarray(metrics["lengths"])
        mean_ret = float(returns[finished].mean()) if finished.any() else float("nan")
        mean_len = float(lengths[finished].mean()) if finished.any() else float("nan")

        # loss / q_value only count iterations where a gradient step actually fired
        # (skipped steps report 0.0 and would otherwise drag the averages toward zero).
        trained = np.asarray(metrics["trained"]).astype(bool)
        loss_arr = np.asarray(metrics["loss"])
        qval_arr = np.asarray(metrics["q_val"])
        mean_loss = float(loss_arr[trained].mean()) if trained.any() else float("nan")
        mean_q = float(qval_arr[trained].mean()) if trained.any() else float("nan")

        eps = float(np.asarray(metrics["epsilon"])[-1])
        sps = int((global_step - prev_step) / max(now - prev_time, 1e-6))  # instantaneous
        prev_step, prev_time = global_step, now

        print(f"step={global_step:>9}  ep_return={mean_ret:8.2f}  ep_len={mean_len:7.1f}  "
              f"loss={mean_loss:8.4f}  q={mean_q:7.3f}  eps={eps:4.2f}  SPS={sps}")
        history.append({
            "global_step": global_step, "episodic_return": mean_ret,
            "episodic_length": mean_len, "loss": mean_loss, "q_value": mean_q,
            "epsilon": eps, "sps": sps,
        })
        if args.track:
            import wandb
            wandb.log({"charts/episodic_return": mean_ret,
                       "charts/episodic_length": mean_len,
                       "losses/loss": mean_loss, "losses/q_values": mean_q,
                       "charts/epsilon": eps, "charts/SPS": sps}, step=global_step)
        if rtpt is not None:
            rtpt.step(subtitle=f"ret={mean_ret:.1f}")

    # ---- save results -------------------------------------------------------------
    if args.save_results:
        out_dir = os.path.join(args.results_dir, ALGO_NAME, game_dir_name(args.game),
                               mode_dir_name(args.obs_mode))
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_{args.run_tag}" if args.run_tag else ""
        tag = f"c51_{args.obs_mode}{suffix}"

        # Wide metrics CSV (all tracked series). The legacy 2-column scores CSV is kept
        # for backward compatibility with existing plotting/report scripts.
        csv_path = os.path.join(out_dir, f"{tag}_metrics.csv")
        with open(csv_path, "w") as f:
            f.write(",".join(COLUMNS) + "\n")
            for row in history:
                f.write(",".join(str(row[c]) for c in COLUMNS) + "\n")
        print(f"metrics -> {csv_path}")

        scores_path = os.path.join(out_dir, f"{tag}_scores.csv")
        with open(scores_path, "w") as f:
            f.write("global_step,episodic_return\n")
            for row in history:
                f.write(f"{row['global_step']},{row['episodic_return']}\n")
        print(f"scores  -> {scores_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = np.array([r["global_step"] for r in history], dtype=np.float64)

            def _series(col):
                return np.array([r[col] for r in history], dtype=np.float64)

            # Multi-panel dashboard (mirrors CleanRL's TensorBoard charts/ + losses/).
            panels = [
                ("episodic_return", "episodic return", "charts"),
                ("episodic_length", "episodic length", "charts"),
                ("loss", "cross-entropy loss", "losses"),
                ("q_value", "mean Q-value", "losses"),
                ("epsilon", "epsilon", "charts"),
                ("sps", "steps / second", "charts"),
            ]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for ax, (col, ylabel, group) in zip(axes.ravel(), panels):
                ys = _series(col)
                mask = ~np.isnan(ys)
                if mask.any():
                    ax.plot(steps[mask], ys[mask])
                ax.set_title(f"{group}/{col}")
                ax.set_xlabel("global step (frames)")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
            fig.suptitle(f"C51 ({args.obs_mode}) — {args.game}")
            fig.tight_layout()
            dash_path = os.path.join(out_dir, f"{tag}_metrics.png")
            fig.savefig(dash_path, dpi=110)
            plt.close(fig)
            print(f"panels  -> {dash_path}")

            # Standalone learning curve (kept for the report / backward compatibility).
            ret = _series("episodic_return")
            m = ~np.isnan(ret)
            if m.any():
                plt.figure()
                plt.plot(steps[m], ret[m])
                plt.xlabel("global step (frames)")
                plt.ylabel("episodic return")
                plt.title(f"C51 ({args.obs_mode}) — {args.game}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                png_path = os.path.join(out_dir, f"{tag}_learning_curve.png")
                plt.savefig(png_path)
                plt.close()
                print(f"curve   -> {png_path}")
        except Exception as e:
            print(f"(skipped plots: {e})")

    return carry


if __name__ == "__main__":
    main(tyro.cli(Args))
