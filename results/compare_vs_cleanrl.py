"""
Learning-curve comparison: our C51 vs the CleanRL reference on Pong (episodic return).

Group 27, Topic 27 (JAXAtari baselines).

The valid reward comparison is OUR PIXEL (RGB) run vs CleanRL's pixel C51 on Atari Pong:
same game, same +/-21 reward scale, so episodic return is directly comparable (the
JAXAtari-vs-gym backend difference matters for SPEED, not reward). Object-centric is
overlaid only as our own secondary curve (CleanRL has no OC baseline).

CleanRL reference: extracted from the official run's tensorboard log
(huggingface.co/cleanrl/PongNoFrameskip-v4-c51_atari_jax-seed1) into
results/pong/cleanrl_c51_pong_seed1_metrics.csv (long format: tag,global_step,value).

Usage:
    python results/compare_vs_cleanrl.py            # ours (pixel+OC) vs CleanRL curve
    python results/compare_vs_cleanrl.py --no-oc    # pixel vs CleanRL only

Output: results/pong/c51_vs_cleanrl.png
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


def read_scores(path):
    """Read a 2-column (global_step, episodic_return) CSV; skip NaN rows."""
    xs, ys = [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            try:
                x = float(row[0]); y = float(row[1])
            except ValueError:
                continue
            if y == y:
                xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def read_cleanrl_tag(path, tag):
    """Read (global_step, value) for one tag from the long CleanRL CSV."""
    xs, ys = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("tag") != tag:
                continue
            try:
                xs.append(float(row["global_step"])); ys.append(float(row["value"]))
            except (ValueError, TypeError):
                continue
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


def smooth(ys, window):
    """Moving average with edge padding (no end-of-curve dip from zero-padding)."""
    if window <= 1 or len(ys) < window:
        return ys
    lpad, rpad = window // 2, window - 1 - window // 2
    yp = np.pad(ys, (lpad, rpad), mode="edge")
    return np.convolve(yp, np.ones(window) / window, mode="valid")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="pong")
    p.add_argument("--ours-pixel-csv",
                   default=str(RESULTS_DIR / "c51_pixel_scores.csv"))
    p.add_argument("--ours-oc-csv",
                   default=str(RESULTS_DIR / "c51_object_centric_scores.csv"))
    p.add_argument("--cleanrl-csv", default=str(DEFAULT_CLEANRL))
    p.add_argument("--no-oc", action="store_true")
    p.add_argument("--out", default=str(RESULTS_DIR / "c51_vs_cleanrl.png"))
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    cx, cy = read_cleanrl_tag(args.cleanrl_csv, "charts/episodic_return")
    if len(cx):
        plt.plot(cx, smooth(cy, max(1, len(cy) // 120)),
                 label="CleanRL c51_atari_jax (ALE, seed 1)",
                 color="C3", linewidth=1.8, linestyle="--")

    px, py = read_scores(args.ours_pixel_csv)
    plt.plot(px, py, label="Ours: pixel (JAXAtari)", color="C0", linewidth=1.8)

    if not args.no_oc and os.path.exists(args.ours_oc_csv):
        ox, oy = read_scores(args.ours_oc_csv)
        plt.plot(ox, oy, label="Ours: object-centric (JAXAtari)",
                 color="C2", linewidth=1.4, alpha=0.9)

    plt.xlabel("global step (frames)")
    plt.ylabel("episodic return")
    plt.title(f"C51 on {args.game}: ours vs CleanRL")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f"comparison plot -> {args.out}")


if __name__ == "__main__":
    main()
