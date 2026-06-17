# Project Context: JAXtari RL Baselines

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

## What we are NOT doing

- We are not building a multi-file framework — each agent is standalone
- We are not using PyTorch — everything is JAX
- We are not required to solve hard games like MontezumaRevenge — just get training running
