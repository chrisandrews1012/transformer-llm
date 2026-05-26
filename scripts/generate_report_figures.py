"""
Generates experiment figures for the training report.
"""

from pathlib import Path
from typing import Callable, List

import matplotlib.pyplot as plt

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "tiny": "#4C72B0",
    "small": "#55A868",
    "tiny_postnorm": "#C44E52",
    "tiny_rope": "#8172B2",
    "tiny_rmsnorm": "#CCB974",
}

def save(name: str) -> None:
    """
    Apply tight layout, save the current figure, and close it.

    :param name: Output filename relative to ``OUTPUT_DIR``.
    :type name: str
    """
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=150)
    plt.close()
    print(f"Saved {name}")


def bar_chart(
    labels: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    bar_colors: List[str],
    label_fmt: Callable[[float], str],
    filename: str,
    label_offset: float,
) -> None:
    """
    Draw a bar chart and save it to disk.

    :param labels: Category labels for each bar.
    :type labels: List[str]
    :param values: Numeric value for each bar.
    :type values: List[float]
    :param ylabel: Y-axis label.
    :type ylabel: str
    :param title: Chart title.
    :type title: str
    :param bar_colors: Fill color for each bar.
    :type bar_colors: List[str]
    :param label_fmt: Function that formats a float into the bar-top annotation string.
    :type label_fmt: Callable[[float], str]
    :param filename: Output filename passed to :func:`save`.
    :type filename: str
    :param label_offset: Vertical gap between bar top and annotation text.
    :type label_offset: float
    """
    _, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=bar_colors, width=0.5, edgecolor="white")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + label_offset,
                label_fmt(val), ha="center", va="bottom", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(filename)

# Experiment 1: Model Size (tiny vs small)
exp1_labels = ["tiny", "small"]
exp1_val_loss   = [3.978, 3.427]
exp1_train_time = [342,   810]
exp1_params     = [5.3,   13.8]
exp1_colors     = [COLORS["tiny"], COLORS["small"]]

bar_chart(
    exp1_labels, exp1_val_loss,
    ylabel="Validation Loss",
    title="Validation Loss by Model Size",
    bar_colors=exp1_colors,
    label_fmt=lambda v: f"{v:.3f}",
    filename="experiment1_val_loss.png",
    label_offset=0.05,
)

bar_chart(
    exp1_labels, exp1_train_time,
    ylabel="Training Time (seconds)",
    title="Training Time by Model Size",
    bar_colors=exp1_colors,
    label_fmt=lambda v: f"{v}s",
    filename="experiment1_train_time.png",
    label_offset=10,
)

# Scatter: parameter count vs val loss
_, ax = plt.subplots(figsize=(6, 4))
ax.scatter(exp1_params, exp1_val_loss, color=exp1_colors, s=120, zorder=3)
for i, label in enumerate(["tiny", "small"]):
    ax.annotate(label, (exp1_params[i], exp1_val_loss[i]),
                textcoords="offset points", xytext=(8, 0), fontsize=10)
ax.set_xlabel("Parameter Count (millions)")
ax.set_ylabel("Validation Loss")
ax.set_title("Parameter Count vs Validation Loss")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save("experiment1_param_vs_loss.png")

# Experiment 2: Pre-norm vs Post-norm (tiny vs tiny_postnorm)
exp2_labels = ["pre-norm", "post-norm"]
exp2_val_loss   = [3.978, 4.119]
exp2_train_time = [342,   364]
exp2_colors     = [COLORS["tiny"], COLORS["tiny_postnorm"]]

bar_chart(
    exp2_labels, exp2_val_loss,
    ylabel="Validation Loss",
    title="Validation Loss by Norm Placement",
    bar_colors=exp2_colors,
    label_fmt=lambda v: f"{v:.3f}",
    filename="experiment2_val_loss.png",
    label_offset=0.05,
)

bar_chart(
    exp2_labels, exp2_train_time,
    ylabel="Training Time (seconds)",
    title="Training Time by Norm Placement",
    bar_colors=exp2_colors,
    label_fmt=lambda v: f"{v}s",
    filename="experiment2_train_time.png",
    label_offset=5,
)

# Experiment 3: Sinusoidal vs RoPE (tiny vs tiny_rope)
exp3_labels     = ["sinusoidal", "rope"]
exp3_val_loss   = [3.978, 3.018]
exp3_train_time = [342,   353]
exp3_colors     = [COLORS["tiny"], COLORS["tiny_rope"]]

bar_chart(
    exp3_labels, exp3_val_loss,
    ylabel="Validation Loss",
    title="Validation Loss by Positional Encoding",
    bar_colors=exp3_colors,
    label_fmt=lambda v: f"{v:.3f}",
    filename="experiment3_val_loss.png",
    label_offset=0.05,
)

bar_chart(
    exp3_labels, exp3_train_time,
    ylabel="Training Time (seconds)",
    title="Training Time by Positional Encoding",
    bar_colors=exp3_colors,
    label_fmt=lambda v: f"{v}s",
    filename="experiment3_train_time.png",
    label_offset=5,
)

# Experiment 4: LayerNorm vs RMSNorm (tiny vs tiny_rmsnorm)
exp4_labels     = ["layernorm", "rmsnorm"]
exp4_val_loss   = [3.978, 4.009]
exp4_train_time = [342,   342]
exp4_colors     = [COLORS["tiny"], COLORS["tiny_rmsnorm"]]

bar_chart(
    exp4_labels, exp4_val_loss,
    ylabel="Validation Loss",
    title="Validation Loss by Norm Type",
    bar_colors=exp4_colors,
    label_fmt=lambda v: f"{v:.3f}",
    filename="experiment4_val_loss.png",
    label_offset=0.05,
)

bar_chart(
    exp4_labels, exp4_train_time,
    ylabel="Training Time (seconds)",
    title="Training Time by Norm Type",
    bar_colors=exp4_colors,
    label_fmt=lambda v: f"{v}s",
    filename="experiment4_train_time.png",
    label_offset=5,
)

