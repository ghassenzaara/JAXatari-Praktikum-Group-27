# JAXtari RL Baselines — Topic 27

Classic deep-RL algorithms implemented in **JAX/Flax** and running on
[**JAXAtari**](https://github.com/k4ntz/JAXAtari), the GPU-accelerated, object-centric
Atari suite from TU Darmstadt's AI/ML Lab. University praktikum project.

Each agent is a **single self-contained file** in `agents/` (network, replay buffer,
training loop, hyperparameters, env setup all in one place — CleanRL style) and supports
**two observation modes**: raw pixels (CNN) and object-centric features (MLP).

| Algorithm | File | Status |
|---|---|---|
| C51 (Categorical DQN) | `agents/c51_jaxtari.py` | ✅ runs (pixel + object-centric) |
| IQN | `agents/iqn_jaxtari.py` | ⬜ planned |
| Rainbow | `agents/rainbow_jaxtari.py` | ⬜ planned |
| IMPALA | `agents/impala_jaxtari.py` | ⬜ planned |

---

## Requirements

- **Linux or WSL2** (Ubuntu). On Windows, run everything inside WSL — not native Windows.
- **NVIDIA GPU** with CUDA 12 drivers (the project trains on GPU).
- **Python 3.12**

## Setup

```bash
# 1. clone
git clone <your-repo-url> Praktikum_JAXatari
cd Praktikum_JAXatari

# 2. virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. core deps
pip install -U "jax[cuda12]"
pip install flax optax tyro matplotlib

# 4. the environment suite + game sprites
pip install git+https://github.com/k4ntz/JAXAtari.git
install-sprites          # downloads Atari sprites (required by jaxatari.make)

# 5. verify the GPU is visible
python -c "import jax; print(jax.devices())"
# expected: [CudaDevice(id=0)]
```

> If you have the repo's `requirements.txt`, `pip install -r requirements.txt` covers steps 3–4.

## Running C51

```bash
# pixel observations (CNN)
python agents/c51_jaxtari.py --game pong --obs-mode pixel --total-timesteps 200000

# object-centric observations (MLP)
python agents/c51_jaxtari.py --game pong --obs-mode object_centric --total-timesteps 200000
```

Useful flags (see the top of the file for the full list):

| Flag | Default | Notes |
|---|---|---|
| `--game` | `pong` | any JAXAtari game (pong, seaquest, breakout, freeway, …) |
| `--obs-mode` | `pixel` | `pixel` (CNN) or `object_centric` (MLP) |
| `--num-envs` | `8` | parallel environments (vmapped) |
| `--total-timesteps` | `10000000` | total env frames; use ~200k for a quick test |
| `--buffer-size` | auto | replay capacity; defaults to 10k (pixel) / 100k (OC) |
| `--learning-starts` | `80000` | frames before training begins (lower for quick tests) |
| `--optimizer` | `adam` | `adam` (CleanRL baseline) or `rmsprop` (original paper) |

## Outputs

After a run you get, under `results/<game>/`:

- `c51_<mode>_scores.csv` — episodic return vs. training step
- `c51_<mode>_learning_curve.png` — the learning curve

## Notes / gotchas

- **GPU memory.** The replay buffer lives on the GPU (it sits inside `jax.lax.scan`), so
  pixel buffers are VRAM-bound. Defaults are small on purpose (10k pixel / 100k OC). On an
  8 GB card keep pixel ≤ ~10k; on larger cards you can raise `--buffer-size`. This is a
  known, supervisor-approved trade-off (smaller buffer than the paper's 1M).
- **First launch is slow.** The whole training loop is JIT-compiled on the first chunk, so
  the first `SPS` reading is low — it climbs afterward.
- **`nan` return / `0` loss early on** is normal: returns show `nan` until the first episode
  finishes, and loss stays 0 until `--learning-starts` frames have passed.

## Project layout

```
agents/        single-file agents (one per algorithm)
results/       scores + learning curves per game
C51 doccumentation.md   consolidated CleanRL C51 reference used to build the agent
CLAUDE.md      project context, design decisions, supervisor Q&A
```

## Credits

- **Algorithm:** Bellemare, Dabney & Munos (2017), *A Distributional Perspective on RL* —
  https://arxiv.org/abs/1707.06887
- **Reference implementation:** [CleanRL](https://github.com/vwxyzjn/cleanrl)
  (`c51_atari_jax.py`, `c51_jax.py`), MIT-licensed.
- **Environment:** [JAXAtari](https://github.com/k4ntz/JAXAtari), TU Darmstadt AI/ML Lab.
