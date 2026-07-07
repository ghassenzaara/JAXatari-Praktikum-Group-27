"""
6-panel metric comparison: our C51 (pixel + object-centric) vs CleanRL on Pong.

Group 27, Topic 27 (JAXAtari baselines).

One figure, 6 panels (episodic return, episodic length, loss, q-values, epsilon, SPS);
each panel overlays up to THREE curves so all versions compare at a glance:
  - CleanRL c51_atari_jax  (reference, ALE = Arcade Learning Environment / original Atari)
  - Ours: pixel   (JAXAtari, CNN)
  - Ours: object-centric   (JAXAtari, MLP)

Inputs
------
CleanRL: results/pong/cleanrl_c51_pong_seed1_metrics.csv  (long: tag,global_step,value),
  extracted from the official tensorboard log
  (huggingface.co/cleanrl/PongNoFrameskip-v4-c51_atari_jax-seed1).
Ours: results/pong/c51_<mode>_metrics.csv  (wide: global_step,episodic_return,
  episodic_length,loss,q_value,epsilon,sps), written by the current c51_jaxtari.py.
  Any of our sources that is missing is simply skipped (with a warning). NOTE the original
  pixel run predates metrics logging, so the pixel curve appears only after a rerun:
      python agents/c51_jaxtari.py --game pong --obs-mode pixel --total-timesteps 10000000

Caveats to note in the report
-----------------------------
- SPS is not apples-to-apples: CleanRL runs 1 CPU gym env, ours runs num_envs GPU envs.
- x-axis is environment frames for both; ours sums frames across parallel envs.
- episodic length is counted differently by JAXAtari vs ALE (note, not a bug).

Usage:
    python results/compare_metrics_vs_cleanrl.py         # CleanRL + pixel + OC (whatever exists)
Output: results/pong/c51_all_vs_cleanrl_metrics.png
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Results layout: results/<ALGO>/<Game>/<files> (e.g. results/C51/Pong/...).
RESULTS_DIR = REPO_ROOT / "results" / "C51" / "Pong"
DEFAULT_CLEANRL = RESULTS_DIR / "cleanrl_c51_pong_seed1_metrics.csv"

# (panel title, CleanRL tag, our metrics.csv column)
PANELS = [
    ("episodic return", "charts/episodic_return", "episodic_return"),
    ("episodic length", "charts/episodic_length", "episodic_length"),
    ("loss",            "losses/loss",            "loss"),
    ("q-values",        "losses/q_values",        "q_value"),
    ("epsilon",         "charts/epsilon",         "epsilon"),
    ("SPS",             "charts/SPS",             "sps"),
]


def smooth(ys, window):
    """Moving average with edge padding (no end-of-curve dip from zero-padding)."""
    ys = np.asarray(ys, dtype=float)
    if window <= 1 or len(ys) < window:
        return ys
    lpad, rpad = window // 2, window - 1 - window // 2
    yp = np.pad(ys, (lpad, rpad), mode="edge")
    return np.convolve(yp, np.ones(window) / window, mode="valid")


def read_cleanrl(path):
    """Return {tag: (steps, values)} from the long CleanRL CSV."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            tag = row.get("tag")
            try:
                x = float(row["global_step"]); y = float(row["value"])
            except (ValueError, TypeError, KeyError):
                continue
            data.setdefault(tag, ([], []))
            data[tag][0].append(x); data[tag][1].append(y)
    out = {}
    for tag, (xs, ys) in data.items():
        order = np.argsort(xs)
        out[tag] = (np.array(xs)[order], np.array(ys)[order])
    return out


def read_ours(path):
    """Return {column: (steps, values)} from our wide metrics CSV (NaNs dropped), or None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        r = csv.DictReader(f)
        names = [c for c in r.fieldnames if c != "global_step"]
        series = {c: ([], []) for c in names}
        for row in r:
            try:
                gs = float(row["global_step"])
            except (ValueError, TypeError, KeyError):
                continue
            for c in names:
                try:
                    v = float(row[c])
                except (ValueError, TypeError):
                    continue
                if v == v:  # not NaN
                    series[c][0].append(gs); series[c][1].append(v)
    return {c: (np.array(xs), np.array(ys)) for c, (xs, ys) in series.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="pong")
    p.add_argument("--ours-pixel-csv",
                   default=str(RESULTS_DIR / "c51_pixel_metrics.csv"))
    p.add_argument("--ours-oc-csv",
                   default=str(RESULTS_DIR / "c51_object_centric_metrics.csv"))
    p.add_argument("--cleanrl-csv", default=str(DEFAULT_CLEANRL))
    p.add_argument("--out",
                   default=str(RESULTS_DIR / "c51_all_vs_cleanrl_metrics.png"))
    args = p.parse_args()

    cleanrl = read_cleanrl(args.cleanrl_csv)

    # Our sources: (label, color, {column: (steps, values)} or None)
    ours_sources = [
        ("Ours: pixel", "C0", read_ours(args.ours_pixel_csv)),
        ("Ours: object-centric", "C2", read_ours(args.ours_oc_csv)),
    ]
    for (label, _, cols), path in zip(ours_sources, [args.ours_pixel_csv, args.ours_oc_csv]):
        if cols is None:
            print(f"WARNING: {label} metrics not found ({path}) - skipping that curve. "
                  f"Re-run that mode with the current c51_jaxtari.py to include it.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (title, tag, col) in zip(axes.ravel(), PANELS):
        if tag in cleanrl:
            cx, cy = cleanrl[tag]
            ax.plot(cx, smooth(cy, max(1, len(cy) // 150)),
                    color="C3", linestyle="--", linewidth=1.6, label="CleanRL (ALE)")
        for label, color, cols in ours_sources:
            if cols is not None and col in cols and len(cols[col][0]):
                ox, oy = cols[col]
                ax.plot(ox, oy, color=color, linewidth=1.6, label=label)
        ax.set_title(title)
        ax.set_xlabel("frames")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"C51 on {args.game}: ours (pixel + object-centric) vs CleanRL c51_atari_jax",
                 y=1.00)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"3-way 6-panel comparison -> {args.out}")


if __name__ == "__main__":
    main()
