"""
6-panel metric comparison: our agent (pixel + object-centric) vs CleanRL.

Group 27, Topic 27 (JAXAtari baselines).

Generalized over algorithm and game via --algo/--game (default c51/pong). Paths are derived
from results/<Algo>/<Game>/ using the lowercase algo key as the filename prefix (c51_, iqn_,
...), matching what the agents write; pass explicit --*-csv/--out to override.

One figure, 6 panels (episodic return, episodic length, loss, q-values, epsilon, SPS);
each panel overlays up to THREE curves so all versions compare at a glance:
  - CleanRL  (reference, ALE = Arcade Learning Environment / original Atari)
  - Ours: pixel   (JAXAtari, CNN)
  - Ours: object-centric   (JAXAtari, MLP)

Inputs
------
CleanRL: results/<Algo>/<Game>/cleanrl_<algo>_<game>_seed<seed>_metrics.csv  (long:
  tag,global_step,value), extracted from the official tensorboard log (e.g. C51/Pong:
  huggingface.co/cleanrl/PongNoFrameskip-v4-c51_atari_jax-seed1).
Ours: results/<Algo>/<Game>/<algo>_<mode>_metrics.csv  (wide: global_step,episodic_return,
  episodic_length,loss,q_value,epsilon,sps), written by the agent. Any missing source is
  simply skipped (with a warning).

Caveats to note in the report
-----------------------------
- SPS is not apples-to-apples: CleanRL runs 1 CPU gym env, ours runs num_envs GPU envs.
- x-axis is environment frames for both; ours sums frames across parallel envs.
- episodic length is counted differently by JAXAtari vs ALE (note, not a bug).

Usage:
    python results/compare_metrics_vs_cleanrl.py                       # c51 / pong (default)
    python results/compare_metrics_vs_cleanrl.py --algo iqn --game frostbite
Layout: reads results/<Algo>/<Game>/{Pixel,ObjectCentric,CleanRL}/, writes the figure into
the sibling Report/ folder (next to the .tex that embeds it).
Output: results/<Algo>/<Game>/Report/<algo>_all_vs_cleanrl_metrics.png
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
    p.add_argument("--algo", default="c51", help="algorithm key (c51, iqn, rainbow, impala)")
    p.add_argument("--game", default="pong", help="game key (pong, frostbite, seaquest, ...)")
    p.add_argument("--seed", type=int, default=1,
                   help="seed used in the CleanRL reference filename")
    # Paths below default to None and are derived from --algo/--game; pass to override.
    p.add_argument("--ours-pixel-csv", default=None)
    p.add_argument("--ours-oc-csv", default=None)
    p.add_argument("--cleanrl-csv", default=None,
                   help="CleanRL reference CSV; default cleanrl_<algo>_<game>_seed<seed>_metrics.csv")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    algo, game = args.algo.lower(), args.game.lower()
    rdir = results_dir(algo, game)
    ours_pixel = args.ours_pixel_csv or str(rdir / "Pixel" / f"{algo}_pixel_metrics.csv")
    ours_oc = args.ours_oc_csv or str(rdir / "ObjectCentric" / f"{algo}_object_centric_metrics.csv")
    cleanrl_csv = args.cleanrl_csv or str(
        rdir / "CleanRL" / f"cleanrl_{algo}_{game}_seed{args.seed}_metrics.csv")
    out = args.out or str(rdir / "Report" / f"{algo}_all_vs_cleanrl_metrics.png")
    algo_title = ALGO_DISPLAY.get(algo, algo.upper())

    cleanrl = read_cleanrl(cleanrl_csv) if os.path.exists(cleanrl_csv) else {}
    if not cleanrl:
        print(f"WARNING: no CleanRL reference at {cleanrl_csv} - CleanRL curves skipped.")

    # Our sources: (label, color, {column: (steps, values)} or None)
    ours_sources = [
        ("Ours: pixel", "C0", read_ours(ours_pixel)),
        ("Ours: object-centric", "C2", read_ours(ours_oc)),
    ]
    for (label, _, cols), path in zip(ours_sources, [ours_pixel, ours_oc]):
        if cols is None:
            print(f"WARNING: {label} metrics not found ({path}) - skipping that curve. "
                  f"Re-run that mode with the current agent to include it.")

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
    fig.suptitle(f"{algo_title} on {game}: ours (pixel + object-centric) vs CleanRL",
                 y=1.00)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"3-way 6-panel comparison -> {out}")


if __name__ == "__main__":
    main()
