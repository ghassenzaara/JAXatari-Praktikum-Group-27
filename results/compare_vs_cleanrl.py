"""
Learning-curve comparison: our agent vs the CleanRL reference (episodic return).

Group 27, Topic 27 (JAXAtari baselines).

Generalized over algorithm and game via --algo/--game (default c51/pong). All paths are
derived from results/<Algo>/<Game>/ using the lowercase algo key as the filename prefix
(c51_, iqn_, ...), matching what the agents write; pass explicit --*-csv/--out to override.

The valid reward comparison is OUR PIXEL (RGB) run vs CleanRL's pixel run on the same game
and reward scale, so episodic return is directly comparable (the JAXAtari-vs-gym backend
difference matters for SPEED, not reward). Object-centric is overlaid only as our own
secondary curve (CleanRL has no OC baseline).

CleanRL reference: extracted from the official run's tensorboard log into
results/<Algo>/<Game>/cleanrl_<algo>_<game>_seed<seed>_metrics.csv (long: tag,global_step,
value). E.g. for C51/Pong: huggingface.co/cleanrl/PongNoFrameskip-v4-c51_atari_jax-seed1.

Usage:
    python results/compare_vs_cleanrl.py                          # c51 / pong (default)
    python results/compare_vs_cleanrl.py --algo iqn --game frostbite
    python results/compare_vs_cleanrl.py --no-oc                  # pixel vs CleanRL only

Layout: reads results/<Algo>/<Game>/{Pixel,ObjectCentric,CleanRL}/, writes the figure into
the sibling Report/ folder (next to the .tex that embeds it).
Output: results/<Algo>/<Game>/Report/<algo>_vs_cleanrl.png
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Results layout: results/<Algo>/<Game>/<files> (e.g. results/C51/Pong/...). Folder names
# are TitleCase; filenames use the lowercase algo key as prefix (c51_, iqn_, ...), matching
# what the agents write. These maps mirror the agents' own display maps.
ALGO_DISPLAY = {"c51": "C51", "iqn": "IQN", "rainbow": "Rainbow", "impala": "IMPALA"}
GAME_DISPLAY = {"mspacman": "MsPacman", "montezumarevenge": "MontezumaRevenge",
                "beamrider": "BeamRider"}


def results_dir(algo, game):
    """results/<Algo>/<Game>/ for a lowercase (algo, game) key pair."""
    return (REPO_ROOT / "results" / ALGO_DISPLAY.get(algo, algo.upper())
            / GAME_DISPLAY.get(game, game.capitalize()))


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
    p.add_argument("--algo", default="c51", help="algorithm key (c51, iqn, rainbow, impala)")
    p.add_argument("--game", default="pong", help="game key (pong, frostbite, seaquest, ...)")
    p.add_argument("--seed", type=int, default=1,
                   help="seed used in the CleanRL reference filename")
    # Paths below default to None and are derived from --algo/--game; pass to override.
    p.add_argument("--ours-pixel-csv", default=None)
    p.add_argument("--ours-oc-csv", default=None)
    p.add_argument("--cleanrl-csv", default=None,
                   help="CleanRL reference CSV; default cleanrl_<algo>_<game>_seed<seed>_metrics.csv")
    p.add_argument("--cleanrl-label", default=None,
                   help="legend label for the CleanRL curve (default: 'CleanRL (ALE, seed <seed>)')")
    p.add_argument("--no-oc", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    algo, game = args.algo.lower(), args.game.lower()
    rdir = results_dir(algo, game)
    ours_pixel = args.ours_pixel_csv or str(rdir / "Pixel" / f"{algo}_pixel_scores.csv")
    ours_oc = args.ours_oc_csv or str(rdir / "ObjectCentric" / f"{algo}_object_centric_scores.csv")
    cleanrl_csv = args.cleanrl_csv or str(
        rdir / "CleanRL" / f"cleanrl_{algo}_{game}_seed{args.seed}_metrics.csv")
    cleanrl_label = args.cleanrl_label or f"CleanRL {algo} (ALE, seed {args.seed})"
    out = args.out or str(rdir / "Report" / f"{algo}_vs_cleanrl.png")
    algo_title = ALGO_DISPLAY.get(algo, algo.upper())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    if os.path.exists(cleanrl_csv):
        cx, cy = read_cleanrl_tag(cleanrl_csv, "charts/episodic_return")
        if len(cx):
            plt.plot(cx, smooth(cy, max(1, len(cy) // 120)),
                     label=cleanrl_label, color="C3", linewidth=1.8, linestyle="--")
    else:
        print(f"WARNING: no CleanRL reference at {cleanrl_csv} - skipping that curve.")

    if os.path.exists(ours_pixel):
        px, py = read_scores(ours_pixel)
        plt.plot(px, py, label="Ours: pixel (JAXAtari)", color="C0", linewidth=1.8)
    else:
        print(f"WARNING: no pixel scores at {ours_pixel} - skipping that curve.")

    if not args.no_oc and os.path.exists(ours_oc):
        ox, oy = read_scores(ours_oc)
        plt.plot(ox, oy, label="Ours: object-centric (JAXAtari)",
                 color="C2", linewidth=1.4, alpha=0.9)

    plt.xlabel("global step (frames)")
    plt.ylabel("episodic return")
    plt.title(f"{algo_title} on {game}: ours vs CleanRL")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"comparison plot -> {out}")


if __name__ == "__main__":
    main()
