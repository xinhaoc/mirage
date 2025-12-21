import matplotlib.pyplot as plt
import numpy as np
import csv

# Colors (same theme as before)
c1 = (31/255, 119/255, 180/255)   # blue  → MPK with overlap
c2 = (174/255.0, 199/255.0, 232/255.0)
c3 = (1, 127/255, 14/255)         # orange/green → MPK without overlap
c4 = (1, 187/255.0, 120/255.0)

batch_sizes = []
mpk_with = []     # mpk+no (with overlap)
mpk_without = []  # mpk_wo  (without overlap)

# -----------------------------------------
# Load CSV
# -----------------------------------------
with open("ab_multi_gpu_new.csv", "r") as f:
    data = csv.reader(f)
    for row in data:
        model = row[0]
        bs = int(row[1])
        system = row[2]
        perf = float(row[3])

        if system == "mpk+no":
            batch_sizes.append(bs)
            mpk_without.append(perf)
        elif system == "mpk_wo":
            mpk_with.append(perf)

batch_sizes = np.array(batch_sizes)
mpk_with = np.array(mpk_with)
mpk_without = np.array(mpk_without)

# Speedup for annotation
speedup = mpk_without / mpk_with

# -----------------------------------------
# Figure: absolute performance
# -----------------------------------------
fig, ax = plt.subplots(
    figsize=(7.5, 3.5),
    constrained_layout=True
)

x = np.arange(len(batch_sizes))
width = 0.35

print("mpk_without:", mpk_without)
print("mpk_with:", mpk_with)

# Absolute performance bars
b0 = ax.bar(
    x - width/2, mpk_without * 1000, width,
    label="MPK (without overlap)", color=c2
)
b1 = ax.bar(
    x + width/2, mpk_with * 1000, width,
    label="MPK (with overlap)", color=c3
)

# -----------------------------------------
# Annotate SPEEDUP above with-overlap bars
# -----------------------------------------

y_top = ax.get_ylim()[1] * 1.15
for rect, sp in zip(b1, speedup):
    ax.text(
        rect.get_x() + rect.get_width()/2,
        y_top,
        f"{sp:.1f}×",
        ha="center",
        va="bottom",
        fontsize=15,
        color=c3
    )

# -----------------------------------------
# Axis labels + title
# -----------------------------------------
ax.set_xticks(x)
ax.set_xticklabels([f"BS={bs}" for bs in batch_sizes], fontsize=15)
ax.set_ylabel("Per-iteration Runtime (us)", fontsize=15, fontweight='bold')

# Add headroom
ax.relim()
ax.autoscale()
ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax * 1.15)

# -----------------------------------------
# Legend (correctly closed)
# -----------------------------------------
fig.legend(
    [b0, b1],
    ["MPK (without overlap)", "MPK (with overlap)"],
    loc="upper center",
    ncol=2,
    fontsize=14,
    bbox_to_anchor=(0.5, 1.15)
)

# Save
plt.savefig("benchmark_ab_multi.pdf", dpi=120, bbox_inches="tight")
plt.close()