# Project Context: JAXtari RL Baselines

## Standing instruction for Claude

**Keep this file current.** Whenever a task surfaces information that will be useful for future work on this project — design decisions, supervisor Q&A, gotchas, repo facts, conventions, paths — record it in this CLAUDE.md so it persists across sessions. Update existing sections rather than duplicating.

## What this project is

This is a university praktikum (TU Darmstadt) where we implement classic deep RL algorithms in JAX and get them running on JAXAtari — a GPU-accelerated, object-centric Atari environment suite built at TU Darmstadt's AI/ML Lab.

**Repository:** https://github.com/k4ntz/JAXAtari

## Our assigned algorithms (Topic 26)

- **C51** (Categorical DQN)
- **IQN** (Implicit Quantile Networks)
- **Rainbow**
- **IMPALA**

## Target environments (JAXtari-15)

Asteroids, Beamrider, Breakout, Enduro, Freeway, Frostbite, Gravitar, Kangaroo, MontezumaRevenge, MsPacman, Phoenix, Pong, Seaquest, Skiing, Tennis.

"Working" = training can start and run. Does not mean solving the game.
Recommended order: **Pong first** (fast, simple), then **Seaquest**.

## Tech stack

| Layer | Library |
|---|---|
| Game environment | JAXAtari (`git+https://github.com/k4ntz/JAXAtari.git`) |
| GPU math + autodiff | JAX (`jax[cuda12]`) |
| Neural networks | Flax (`flax.linen`) |
| Optimizers | Optax |

## JAXAtari observation modes

JAXAtari supports two observation types via wrappers:

```python
# Pixel-based (raw screen, needs CNN)
from jaxatari.wrappers import AtariWrapper, PixelObsWrapper
env = PixelObsWrapper(AtariWrapper(jaxatari.make("pong")))

# Object-centric (structured features, needs MLP)
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, FlattenObservationWrapper
env = FlattenObservationWrapper(ObjectCentricWrapper(AtariWrapper(jaxatari.make("pong"))))
```

Each agent file must support both modes.

## File architecture

```
jaxtari-baselines/
│
├── agents/
│   ├── c51_jaxtari.py        # single-file C51
│   ├── iqn_jaxtari.py        # single-file IQN
│   ├── rainbow_jaxtari.py    # single-file Rainbow
│   └── impala_jaxtari.py     # single-file IMPALA
│
├── results/
│   └── pong/
│       ├── c51_scores.csv
│       └── c51_learning_curve.png
│
├── requirements.txt
└── README.md
```

## Key constraints

- **One file per algorithm** — everything self-contained (network, training loop, hyperparameters, environment setup)
- **JAX-native** — use `jit`, `vmap`, `lax.scan` where appropriate for GPU efficiency
- **Credit sources** — if reusing an existing JAX implementation, credit the original authors inside the file
- **Both observation modes** — each agent must work with pixel and object-centric input
- **Hyperparameters** — pixel and object-centric modes need separate tuning (different architectures: CNN vs MLP)

## Grading notes

- A final report is required detailing: sources used, who did what, and results vs original paper
- "It runs" is not enough for a 1.0 — quality, report, and comparison to original results matter
- Results to report: learning curves and final scores per game, compared to reference implementations

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -U "jax[cuda12]"
pip install flax optax
pip install git+https://github.com/k4ntz/JAXAtari.git
install-sprites

# Verify GPU is detected
python3 -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]
```

## GPU server access (TU ML Student Pool)

- **Group repo:** https://github.com/ghassenzaara/JAXatari-Praktikum-Group-27 (branch `main`).
- **Studentpool username:** `gzaara`. **SSH key** (ed25519) installed at WSL `~/.ssh/id_ed25519`
  (600) with a matching `~/.ssh/config`; original key lives on Windows at
  `C:\TU\Fs4\AuD Tutor\ssh key.txt` (+`.pub`). Public key was sent to `@raban.emunds`.
- **Usable machines:** mlsp1, mlsp2, mlsp4, mlsp6, mlsp7. Avoid mlsp3/mlsp8 (not in use),
  mlsp5 (GPU driver error). IPs are in `~/.ssh/config`; mlsp1 = 130.83.166.151.
- **Must be on the TU-VPN** to reach the machines — eduroam alone is firewalled off (SSH times
  out). Verified: `ssh gzaara@130.83.166.151` works from Windows PowerShell/cmd once on VPN.
- **WSL quirk:** WSL2 does NOT inherit Windows VPN routes, so `ssh mlsp1` times out from WSL even
  when Windows connects fine. Workarounds: (A) SSH from Windows PowerShell/cmd and use GitHub as
  the code bridge; or (B) enable `networkingMode=mirrored` in `C:\Users\Zaara\.wslconfig` +
  `wsl --shutdown` (needs Win11 22H2 / WSL ≥2.0).
- **Workflow:** develop in WSL → `git push` → `ssh mlsp1` (from PowerShell) → `git pull` on the
  server → run training on the server's GPU. Windows/WSL have no usable GPU; JAX runs on mlsp1.
- **Before experiments (PDF):** use the pip `RTPT` module and join the `mlstudentpool management`
  Mattermost channel. Use a venv (uv) or podman. Back up code to GitHub, checkpoints via rsync.

## What we are NOT doing

- We are not building a multi-file framework — each agent is standalone
- We are not using PyTorch — everything is JAX
- We are not required to solve hard games like MontezumaRevenge — just get training running

---

## Implementation clarifications (from praktikum Q&A)

### 1. Optimizer: RMSProp (original paper) vs Adam (CleanRL baseline)

**Decision: Match the CleanRL baseline — use Adam.**

The original DQN paper uses RMSProp, but CleanRL (our reference implementation) uses Adam, which tends to be a small but reliable improvement. Deviating from the baseline without reason makes comparison harder.

To preserve flexibility, expose the optimizer as a config parameter so users can switch between Adam and RMSProp if they want to reproduce the original paper exactly. Example:

```python
@dataclass
class Config:
    optimizer: str = "adam"  # "adam" or "rmsprop"
```

### 2. Hyperparameters: should pixel and object-centric modes share the same config?

**No — they need separate hyperparameter sets.**

The observation spaces are fundamentally different (raw pixels → CNN; structured features → MLP), so the same config will cause problems. Follow the same pattern as `ppo_oc` vs `ppo_rgb` in the JAXAtari repo: keep hyperparameters as close as possible to the baseline for the default 8-parallel-env runs, but switch the network architecture (CNN → MLP) and tune the relevant HPs accordingly.

The default run uses 8 parallel environments. HP changes between pixel and OC modes should be minimal — only what's necessary to account for the architecture difference.

### 3. Optimized hyperparameter configs: how free are we?

**Goal: demonstrate JAXtari's scalability by scaling up `n_envs`. Keep changes minimal and justified.**

- The main point of an "optimized" config is to show one of JAXtari's core benefits: running many parallel environments on the GPU.
- For **DQN-family algorithms** (C51, IQN, Rainbow): these don't scale favorably in their base form due to the replay buffer / off-policy nature. Scale up `n_envs` as far as possible, but don't be discouraged if training quality degrades — that's expected and worth noting.
- If there is **no meaningful improvement in training time**, you don't need to provide a separate optimized config for that algorithm.
- When scaling `n_envs`, look for papers that describe how other HPs should co-vary. For example, in PPO scaling `n_envs` requires adjusting `num_minibatches` to keep update steps consistent (see: https://arxiv.org/pdf/2603.06009). Find analogous guidance for each algorithm.
- Keep HP changes **minimal** — only change what scaling actually requires.

**PQN as a scalable alternative:**

For algorithms that don't scale in their base DQN-based form, implement a **PQN variant** instead (e.g., `rainbow_pqn.py`). PQN (Parallel Q-Network) is designed for massively parallel environments and is where optimized configs are most valuable. Reference: `scripts/benchmarks/pqn_agent.py` in the JAXAtari repository.

In short:
- Base variant: match the CleanRL baseline as closely as possible.
- Optimized variant: scale `n_envs`, justify every HP change with a paper or clear reasoning, and prefer PQN-based variants for algorithms that don't scale otherwise.

### 4. Replay buffer: CleanRL's ReplayBuffer needs gymnasium Spaces, JAXAtari doesn't provide them

**Decision: extend JAXAtari's spaces — don't bolt on a numpy buffer.**

CleanRL's `ReplayBuffer` requires gymnasium `Space` objects, which JAXAtari doesn't expose. Rather than dropping in a separate numpy ring buffer, add the space we need to JAXAtari directly: create a new class that inherits from JAXAtari's base `Space` class and implement it in JAX, matching the style of the existing jaxtari spaces. If the new space turns out to be commonly used, it can later be upstreamed into JAXAtari's space definitions.

### 5. Per-algorithm reference implementations

- **C51, IMPALA**: adapt from CleanRL's JAX implementations where they exist.
- **Rainbow**: CleanRL only has a PyTorch version (no JAX). Convert the PyTorch implementation to JAX/Flax, following the same pattern as the other JAX agents so it trains under JAX like the CleanRL-JAX algorithms.
- **IMPALA**: there is no CleanRL implementation. Use the original-paper TensorFlow implementation from DeepMind as the baseline: https://github.com/google-deepmind/scalable_agent

### 6. PQN training loop: use `lax.scan`, not Python for-loops

**Decision: use `jax.lax.scan` for the rollout loop whenever possible.**

PQN is rollout-based, like CleanRL's JAX PPO (which uses `lax.scan` for rollout collection). Prefer `scan` over the Python for-loop + `@jax.jit`-on-update pattern from `dqn_atari_jax.py`. Use `scan` wherever it applies across the agents.

### Additional reference implementations (use as inspiration only)

These sometimes have JAX implementations worth studying, but they do **not** follow the CleanRL philosophy we care about (single-file where possible, easy to read, simple configuration). Treat them as inspiration, not templates to copy:

- **acme**: https://github.com/google-deepmind/acme/tree/master/acme/agents/jax
- **dopamine**: https://github.com/google/dopamine/blob/master/dopamine/jax/agents/
- **stoix**: https://github.com/EdanToledo/Stoix/tree/main/stoix/systems

---

## C51 implementation (agents/c51_jaxtari.py)

Status: first full port written (`agents/c51_jaxtari.py`). The old verbatim CleanRL copy
`agents/c51_atari_jax.py` is superseded and should be deleted. A consolidated reference
of the CleanRL sources lives in `C51 doccumentation.md` at the repo root.

### JAXAtari API facts (verified by reading the installed package)

- **Entry point:** `jaxatari.make("pong")` (lowercase names; see `core.GAME_MODULES`). Calls
  `check_ownership()`, so sprites must be installed. Target games are registered (pong,
  seaquest, breakout, freeway, etc.).
- **Base env** (`jaxatari.environment.JaxEnvironment`): `reset(key) -> (obs, state)`;
  `step(state, action) -> (obs, state, reward, done, info)` (5-tuple). Also exposes
  `action_space()`, `observation_space()`, `image_space()`, `render(state)`.
- **AtariWrapper(env, sticky_actions=0.25, episodic_life=True, first_fire=True, noop_max=30,
  full_action_space=False)**: adds sticky actions, episodic-life termination, noop/fire reset.
  `step` now returns a **6-tuple**: `(obs, state, reward, terminated, truncated, info)`.
  Set `sticky_actions=0.0` to match the CleanRL NoFrameskip baseline.
- **PixelObsWrapper(atari, do_pixel_resize=True, pixel_resize_shape=(84,84), grayscale=True,
  frame_stack_size=4, frame_skip=4, max_pooling=True, clip_reward=True)**: returns a stacked
  image. `observation_space()` is a `Box` of shape **(stack, H, W, C)** uint8, e.g.
  `(4, 84, 84, 1)` with grayscale. Frame-skip/max-pool/stack/reward-clip are handled here.
- **ObjectCentricWrapper(atari, frame_stack_size=4, frame_skip=4, clip_reward=True)**: returns
  a `Box` of shape **(stack, num_features)** float32. Wrap with **FlattenObservationWrapper**
  to get a flat **(stack*num_features,)** vector for the MLP.
- **Autoreset is built in** to PixelObsWrapper/ObjectCentricWrapper: on env_done/truncation
  they `lax.cond`-reset the whole stack (gym SAME_STEP mode), advancing the RNG key. The
  rollout loop does NOT need to reset envs manually.
- **LogWrapper(env)**: JAX-native episode logging (replaces gym RecordEpisodeStatistics).
  Adds `info["returned_episode_returns"]`, `["returned_episode_lengths"]`, `["returned_episode"]`.
  Wrap it outermost. Uses unclipped `env_reward` from info for the logged return.
- **Reward** is clipped to sign (+/-1) by the obs wrappers (`clip_reward=True`), so the Atari
  C51 value range `v_min/v_max = -10/10`, `n_atoms=51` is appropriate for **both** modes — no
  need for the 101-atom / +/-100 range from CleanRL's CartPole `c51_jax.py`.

### Design decisions in c51_jaxtari.py

- **One file, two modes** via `--obs-mode {pixel,object_centric}`; CNN vs MLP is the only
  structural difference; all other HPs shared (matches "minimal changes" guidance).
- **Replay buffer:** self-contained jittable JAX buffer (`flax.struct.dataclass`, ring index),
  sized from `env.observation_space().shape/.dtype`. We did NOT subclass `spaces.Space` — the
  wrappers already expose a proper JAXAtari `Box`, so the buffer reads shape/dtype from it
  directly. (If upstreaming a dedicated buffer Space is later wanted, that's the place.)
- **Replay buffer size (supervisor-confirmed):** the buffer lives on the **GPU** because it
  sits inside the `lax.scan` carry. The standard C51 capacity of 1,000,000 pixel frames
  (4x84x84 uint8, obs + next_obs) would need ~56GB of VRAM — impossible on the lab GPUs
  (e.g. RTX 2080 Ti = 11GB; an 8GB card OOMs even at 50k). Raban confirmed on Mattermost
  (2026-06-17): **"small buffer should be fine for now."** So we use mode-aware defaults:
  **pixel 10_000** (tuned 20_000), **object_centric 100_000** (tuned 200_000) — matching the
  team-standard sizes Harsh used. The downside (reduced experience diversity vs. the paper)
  is noted for the report. The considered alternative — a NumPy/CPU buffer with only sampled
  batches moved to GPU — was rejected because it requires breaking out of `lax.scan` for the
  (non-traceable) numpy sampling step. Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` for headroom.
- **Vectorized** with `jax.vmap` over `num_envs` (default 8); the whole rollout+train loop is a
  single `jax.lax.scan`, chunked in Python only to log periodically.
- **C51 core** (greedy-by-expected-value action, distributional projection via `fori_loop`,
  cross-entropy loss) ported verbatim from CleanRL; rewards/dones reshaped to (B,1) for the
  support shift.
- **Optimizer** switch `--optimizer {adam,rmsprop}` (adam default, `eps=0.01/batch_size`).
- **`bash` cannot reach the WSL filesystem** (`\\wsl.localhost\...` is a UNC path the sandbox
  rejects), so training must be run in WSL directly; the Read/Write/Edit tools DO reach it.

### Run commands (execute in WSL, GPU venv)

```bash
python agents/c51_jaxtari.py --game pong --obs-mode pixel         --total-timesteps 200000
python agents/c51_jaxtari.py --game pong --obs-mode object_centric --total-timesteps 200000
```

Outputs `results/<game>/c51_<mode>_scores.csv` and `_learning_curve.png`. CLI flags marked
`# VERIFY` in the file (e.g. `action_space().n`, obs shapes) should be sanity-checked on the
first run.
