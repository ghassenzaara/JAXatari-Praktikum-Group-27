"""Overlay the three OC MLP-width learning curves for C51 on Pong.
512x512 vs 128x64x32 vs 64x32x16. Writes c51_oc_width3_comparison.png.
"""
import pandas as pd
import matplotlib.pyplot as plt

BASE = "results/C51/Pong/ObjectCentric"
runs = [
    ("512x512",   f"{BASE}/c51_object_centric_512x512_scores.csv",   "#2ca02c"),
    ("128x64x32", f"{BASE}/c51_object_centric_128x64x32_scores.csv", "#1f77b4"),
    ("64x32x16",  f"{BASE}/c51_object_centric_64x32x16_scores.csv",  "#ff7f0e"),
]

plt.figure(figsize=(7.0, 4.2))
for label, path, color in runs:
    df = pd.read_csv(path)
    plt.plot(df["global_step"] / 1e6, df["episodic_return"],
             label=f"OC MLP {label}", color=color, linewidth=1.8)

plt.axhline(21, color="gray", linestyle=":", linewidth=0.8)
plt.text(0.1, 21.3, "max +21", color="gray", fontsize=8)
plt.xlabel("Environment frames (millions)")
plt.ylabel("Episodic return")
plt.title("C51 on Pong (object-centric): MLP width comparison")
plt.legend(loc="lower right", frameon=False)
plt.grid(alpha=0.25)
plt.tight_layout()
out = f"{BASE}/../Report/c51_oc_width3_comparison.png"
plt.savefig(out, dpi=150)
print("wrote", out)
