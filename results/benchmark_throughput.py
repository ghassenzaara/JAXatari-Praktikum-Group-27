"""
Throughput benchmark for c51_jaxtari.py - pixel vs object-centric, and num_envs scaling.

Group 27, Topic 27 (JAXAtari baselines).

Purpose
-------
Measure steady-state training throughput (SPS = environment frames per second) of our
C51 agent, so we can (a) compare pixel vs object-centric fairly on the SAME machine, and
(b) show how JAXAtari's GPU-batched envs scale with `num_envs` (the CPU gym baseline
CleanRL runs on cannot do this). This does NOT measure reward - episodic return comes
from the full 10M runs.

How it works
------------
For each (obs_mode, num_envs) it runs agents/c51_jaxtari.py as a subprocess for a short
number of frames, into a throwaway --results-dir, then reads the `sps` column of the
metrics CSV the agent writes. Steady-state SPS = median over the logging chunks after the
first fifth (drops JIT-compile / warm-up chunks). `learning_starts` is set small so the
gradient-step path is exercised (training-phase SPS, not just rollout).

Reusing the real agent as a subprocess means the numbers reflect the actual training loop,
with zero duplicated logic.

Run on the GPU box (JAX must see a CUDA device):
    python results/benchmark_throughput.py                       # pixel vs OC at num_envs=8
    python results/benchmark_throughput.py --num-envs-list 1,8,32,128,256   # scaling scan
    python results/benchmark_throughput.py --modes pixel --num-envs-list 8,64,256

Outputs (in --out-dir, default results/):
    benchmark_throughput.csv   # obs_mode, num_envs, sps, episodic_return(if any), frames
    benchmark_throughput.png   # SPS vs num_envs, one line per mode
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT = REPO_ROOT / "agents" / "c51_jaxtari.py"

# Must mirror c51_jaxtari.py's output layout:
# <results-dir>/<ALGO>/<Game>/<Pixel|ObjectCentric>/<files>.
ALGO_NAME = "C51"
_GAME_DISPLAY = {"mspacman": "MsPacman", "montezumarevenge": "MontezumaRevenge",
                 "beamrider": "BeamRider"}
_MODE_DISPLAY = {"pixel": "Pixel", "object_centric": "ObjectCentric"}


def game_dir_name(game):
    return _GAME_DISPLAY.get(game, game.capitalize())


def mode_dir_name(obs_mode):
    return _MODE_DISPLAY.get(obs_mode, obs_mode)


def run_one(game, obs_mode, num_envs, total_timesteps, learning_starts, num_logs, seed):
    """Run the agent once for (obs_mode, num_envs); return (steady_sps, final_return, frames)."""
    tmp_dir = tempfile.mkdtemp(prefix=f"bench_{obs_mode}_{num_envs}_")
    cmd = [
        sys.executable, str(AGENT),
        "--game", game,
        "--obs-mode", obs_mode,
        "--num-envs", str(num_envs),
        "--total-timesteps", str(total_timesteps),
        "--learning-starts", str(learning_starts),
        "--num-logs", str(num_logs),
        "--seed", str(seed),
        "--results-dir", tmp_dir,
        "--rtpt-initials", "",          # disable RTPT for the benchmark
    ]
    print(f"\n>>> {obs_mode:14s} num_envs={num_envs:<4d} "
          f"({total_timesteps} frames) -> {tmp_dir}")
    t0 = __import__("time").time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    wall = __import__("time").time() - t0
    if proc.returncode != 0:
        print(f"    FAILED (exit {proc.returncode}) - skipping this config")
        return None

    metrics_csv = (Path(tmp_dir) / ALGO_NAME / game_dir_name(game)
                   / mode_dir_name(obs_mode) / f"c51_{obs_mode}_metrics.csv")
    if not metrics_csv.exists():
        print(f"    no metrics csv at {metrics_csv} - skipping")
        return None

    sps_vals, ret_vals = [], []
    with open(metrics_csv) as f:
        for row in csv.DictReader(f):
            try:
                sps_vals.append(float(row["sps"]))
            except (KeyError, ValueError):
                pass
            try:
                r = float(row["episodic_return"])
                if r == r:  # not NaN
                    ret_vals.append(r)
            except (KeyError, ValueError):
                pass

    if not sps_vals:
        print("    no SPS rows - skipping")
        return None

    # Drop the first fifth of chunks (JIT compile + warm-up), take the median of the rest.
    warm = max(1, len(sps_vals) // 5)
    steady = statistics.median(sps_vals[warm:] or sps_vals)
    final_ret = ret_vals[-1] if ret_vals else float("nan")
    print(f"    steady SPS={steady:.0f}   (wall {wall:.1f}s, {len(sps_vals)} chunks, "
          f"warm-up dropped={warm})")
    return steady, final_ret, total_timesteps


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="pong")
    p.add_argument("--modes", default="pixel,object_centric",
                   help="comma-separated obs modes to benchmark")
    p.add_argument("--num-envs-list", default="8",
                   help="comma-separated num_envs values (e.g. 1,8,32,128,256)")
    p.add_argument("--total-timesteps", type=int, default=300_000,
                   help="frames per config; keep short - this measures throughput, not reward")
    p.add_argument("--learning-starts", type=int, default=2_000,
                   help="small so gradient steps fire early (training-phase SPS)")
    p.add_argument("--num-logs", type=int, default=20,
                   help="logging chunks; fewer = larger, less-noisy SPS windows")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    envs = [int(n) for n in args.num_envs_list.split(",") if n.strip()]

    results = []  # (obs_mode, num_envs, sps, final_return, frames)
    for mode in modes:
        for n in envs:
            out = run_one(args.game, mode, n, args.total_timesteps,
                          args.learning_starts, args.num_logs, args.seed)
            if out is not None:
                sps, ret, frames = out
                results.append((mode, n, sps, ret, frames))

    if not results:
        print("\nNo successful runs - nothing to write.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "benchmark_throughput.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obs_mode", "num_envs", "sps", "final_episodic_return", "frames"])
        for row in results:
            w.writerow(row)
    print(f"\nsummary -> {csv_path}")

    # Console table
    print(f"\n{'mode':16s} {'num_envs':>8s} {'SPS':>10s} {'return':>10s}")
    for mode, n, sps, ret, _ in results:
        print(f"{mode:16s} {n:>8d} {sps:>10.0f} {ret:>10.2f}")

    # Plot: SPS vs num_envs, one line per mode (also fine for a single point).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        for mode in modes:
            pts = sorted((n, sps) for m, n, sps, _, _ in results if m == mode)
            if pts:
                xs, ys = zip(*pts)
                plt.plot(xs, ys, marker="o", label=mode)
        plt.xlabel("num_envs (parallel GPU environments)")
        plt.ylabel("steady-state SPS (frames / second)")
        plt.title(f"C51 throughput on {args.game} (JAXAtari)")
        if len(envs) > 1:
            plt.xscale("log", base=2)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        png_path = os.path.join(args.out_dir, "benchmark_throughput.png")
        plt.savefig(png_path, dpi=120)
        print(f"plot    -> {png_path}")
    except Exception as e:
        print(f"(skipped plot: {e})")


if __name__ == "__main__":
    main()
