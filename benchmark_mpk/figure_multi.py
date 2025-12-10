import matplotlib.pyplot as plt
import numpy as np
import csv

# Colors
c1 = (31/255, 119/255, 180/255)
c2 = (174/255, 199/255, 232/255)
c3 = (1, 127/255, 14/255)
c6 = (152/255, 223/255, 138/255)

systems = ["pytorch", "vllm", "sglang", "pytorch+mpk"]
systems_display = ["PyTorch", "vLLM", "SGLang", "MPK"]

batch_sizes = [1,2,4,8,16]
gpu_counts = [1,2,4,8]   # 4 subplots

# -----------------------------------------
# Load CSV
# -----------------------------------------
results = {}  # (gpus, bs, system) → time

with open("multi_gpu.csv", "r") as f:
    data = csv.reader(f)
    for row in data:
        model = row[0]
        gpus = int(row[1])
        bs = int(row[2])
        system = row[3]
        time = float(row[4])
        results[(gpus, bs, system)] = time

# -----------------------------------------
# Label helper
# -----------------------------------------
def autolabel(ax, rects, values, color):
    for rect, v in zip(rects, values):
        ax.text(
            rect.get_x() + rect.get_width()/2,
            rect.get_height(),
            f"{v:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            color=color
        )

# -----------------------------------------
# Create figure: 1 row, 4 subplots
# -----------------------------------------
fig, axes = plt.subplots(
    nrows=1,
    ncols=4,
    figsize=(16, 4),
    constrained_layout=True,
    squeeze=False
)

width = 0.18
x = np.arange(len(batch_sizes))  # 5 batch sizes

# -----------------------------------------
# Draw each GPU subplot
# -----------------------------------------
for col, gpus in enumerate(gpu_counts):
    ax = axes[0][col]

    # Extract times
    pyt = np.array([results[(gpus, bs, "pytorch")] for bs in batch_sizes])
    vll = np.array([results[(gpus, bs, "vllm")] for bs in batch_sizes])
    sgl = np.array([results[(gpus, bs, "sglang")] for bs in batch_sizes])
    mpk = np.array([results[(gpus, bs, "pytorch+mpk")] for bs in batch_sizes])

    baseline = np.minimum(vll, sgl)
    opt = np.minimum(baseline, mpk)

    r_pyt = opt / pyt
    r_vll = opt / vll
    r_sgl = opt / sgl
    r_mpk = opt / mpk

    # 4 bars per batch size
    b0 = ax.bar(x - 1.5*width, r_pyt, width, color=c1)
    b1 = ax.bar(x - 0.5*width, r_vll, width, color=c6)
    b2 = ax.bar(x + 0.5*width, r_sgl, width, color=c2)
    b3 = ax.bar(x + 1.5*width, r_mpk, width, color=c3)

    autolabel(ax, b3, baseline / mpk, c3)

    ax.set_title(f"{gpus} GPUs", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"BS={bs}" for bs in batch_sizes], fontsize=10)
    ax.relim()
    ax.autoscale()

    # Then add a small headroom so annotation text fits
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.10)  # add 10% space above the tallest bar
    ax.tick_params(axis="y", labelsize=10)

# Shared y-axis label
fig.text(0.02, 0.5, "Relative Performance", fontsize=14, rotation="vertical", va="center")

# Shared legend
fig.legend(
    [b0, b1, b2, b3],
    systems_display,
    loc="upper center",
    ncol=4,
    fontsize=12,
    bbox_to_anchor=(0.5, 1.12)
)

plt.savefig("benchmark_multi.pdf", dpi=120, bbox_inches="tight")
plt.close()

# print("benchmark_multi.pdf")
