#!/usr/bin/env python3
"""Run mlptest, parse its stdout, and produce loss + convergence plots.

Output:
  docs/loss.png         - training loss vs iteration (log y-axis)
  docs/convergence.png  - per-sample prediction vs target over iterations
"""

import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
BINARY = REPO / "build" / "mlptest"
OUT_DIR = REPO / "docs"

ITER_RE = re.compile(r"\[(.+?)\]\s*loss\s*=\s*Value\(([-\d.eE+]+)\)")
TARGET_RE = re.compile(r"Target:\[(.+)\]")


def run_training() -> str:
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, check=True)
    return result.stdout


def parse(output: str):
    lines = output.strip().splitlines()
    targets = [float(x) for x in TARGET_RE.match(lines[0]).group(1).split(",")]

    preds_history, losses = [], []
    for line in lines[1:]:
        m = ITER_RE.search(line)
        if not m:
            continue
        preds_history.append([float(x) for x in m.group(1).rstrip(",").split(",")])
        losses.append(float(m.group(2)))
    return targets, preds_history, losses


def plot_loss(losses, path):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(losses, color="#1f77b4", linewidth=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (sum of squared errors)")
    ax.set_title("Training loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(targets, preds_history, path):
    iters = range(len(preds_history))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for j, (color, target) in enumerate(zip(colors, targets)):
        series = [p[j] for p in preds_history]
        ax.plot(iters, series, color=color, linewidth=2, label=f"pred[{j}] (target={target})")
        ax.axhline(target, color=color, linestyle="--", linewidth=1, alpha=0.4)
    ax.set_xlabel("iteration")
    ax.set_ylabel("prediction")
    ax.set_title("Prediction convergence to target (dashed = target)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = run_training()
    targets, preds_history, losses = parse(output)
    plot_loss(losses, OUT_DIR / "loss.png")
    plot_convergence(targets, preds_history, OUT_DIR / "convergence.png")
    print(f"Wrote {OUT_DIR / 'loss.png'} and {OUT_DIR / 'convergence.png'}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Final preds: {preds_history[-1]}")
    print(f"Targets:     {targets}")


if __name__ == "__main__":
    main()
