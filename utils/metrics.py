"""
utils/metrics.py — Metrics, plotting, and results reporting.

Provides:
  • BenchmarkReport: aggregates all results into one object
  • plot_ppl_vs_bits: PPL vs bit-width chart
  • plot_speed_vs_bits: throughput vs bit-width chart
  • plot_memory_vs_bits: memory vs bit-width chart
  • plot_pareto: accuracy–speed Pareto frontier
  • save_results_csv / save_results_json: persist results
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


# ---------------------------------------------------------------------------
# Benchmark Report
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    """Aggregates all benchmark results for a full run."""
    model_name: str
    quant_configs: List[str]          # e.g. ["fp16", "W4G128", "W2G64", "W1G64"]
    ppl_results: Dict[str, float]     # config → PPL
    speed_results: Dict[str, float]   # config → tokens/sec
    memory_results: Dict[str, float]  # config → GB
    compression_results: Dict[str, float]  # config → ratio vs fp16

    def summary_table(self) -> str:
        """Render a combined comparison table."""
        cols = ["Config", "PPL↓", "Tok/s↑", "VRAM(GB)↓", "Compress↑"]
        widths = [18, 10, 10, 12, 12]
        header = "  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
        sep    = "  " + "─" * (sum(widths) + 2 * len(widths))
        lines  = [sep, header, sep]

        for cfg in self.quant_configs:
            ppl   = self.ppl_results.get(cfg, float("nan"))
            speed = self.speed_results.get(cfg, float("nan"))
            mem   = self.memory_results.get(cfg, float("nan"))
            comp  = self.compression_results.get(cfg, float("nan"))
            row   = [cfg,
                     f"{ppl:.3f}", f"{speed:.1f}",
                     f"{mem:.2f}", f"{comp:.2f}×"]
            lines.append(
                "  " + "  ".join(f"{v:<{w}}" for v, w in zip(row, widths))
            )

        lines.append(sep)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "configs": {
                cfg: {
                    "ppl":         self.ppl_results.get(cfg),
                    "tok_per_sec": self.speed_results.get(cfg),
                    "vram_gb":     self.memory_results.get(cfg),
                    "compression": self.compression_results.get(cfg),
                }
                for cfg in self.quant_configs
            }
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _check_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        return plt, ticker
    except ImportError:
        raise ImportError("pip install matplotlib to generate plots")


def plot_ppl_vs_bits(
    bit_widths: List[int],
    ppl_values: List[float],
    label: str = "LLaMA-2-7B",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Line chart: Perplexity (WikiText-2) vs. quantisation bit-width.

    Marks the fp16 baseline with a dashed horizontal line.
    """
    plt, ticker = _check_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bit_widths, ppl_values, "o-", linewidth=2, markersize=7,
            color="#E84A5F", label=label)

    # fp16 baseline (last point assumed to be fp16 if bits=16 present)
    if 16 in bit_widths:
        baseline = ppl_values[bit_widths.index(16)]
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
                   label="fp16 baseline")

    ax.set_xlabel("Quantisation Bits", fontsize=12)
    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=12)
    ax.set_title(f"PPL vs Bit-Width — {label}", fontsize=13, fontweight="bold")
    ax.set_xticks(bit_widths)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_speed_vs_bits(
    bit_widths: List[int],
    throughput_values: List[float],
    label: str = "LLaMA-2-7B",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Bar chart: Tokens/sec vs. bit-width."""
    plt, _ = _check_matplotlib()

    colours = ["#E84A5F", "#FF847C", "#FECEAB", "#99B898", "#2A363B"]
    c = [colours[i % len(colours)] for i in range(len(bit_widths))]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(b) for b in bit_widths], throughput_values,
                  color=c, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, throughput_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xlabel("Quantisation Bits", fontsize=12)
    ax.set_ylabel("Throughput (tokens/second)", fontsize=12)
    ax.set_title(f"Inference Speed vs Bit-Width — {label}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_memory_vs_bits(
    bit_widths: List[int],
    memory_gb: List[float],
    label: str = "LLaMA-2-7B",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Stacked context showing theoretical vs actual memory."""
    plt, _ = _check_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 4))
    theoretical = [7.0 * b / 16.0 * 2 for b in bit_widths]  # 7B model example

    ax.bar([str(b) for b in bit_widths], theoretical, label="Theoretical (weights only)",
           color="#99B898", alpha=0.6)
    ax.bar([str(b) for b in bit_widths], memory_gb, label="Actual (incl. metadata)",
           color="#E84A5F", alpha=0.85)

    ax.set_xlabel("Quantisation Bits", fontsize=12)
    ax.set_ylabel("Memory (GB)", fontsize=12)
    ax.set_title(f"Memory Footprint vs Bit-Width — {label}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_pareto(
    configs: List[str],
    ppl_values: List[float],
    throughput_values: List[float],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Pareto frontier: quality (PPL) vs speed (tokens/sec).

    Lower-left corner = worse on both axes.
    Upper-left (low PPL, high speed) = Pareto-optimal.
    """
    plt, _ = _check_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 5))

    colours = plt.cm.viridis(np.linspace(0.2, 0.9, len(configs)))

    for cfg, ppl, spd, colour in zip(configs, ppl_values, throughput_values, colours):
        ax.scatter(spd, ppl, s=120, color=colour, zorder=5)
        ax.annotate(cfg, (spd, ppl),
                    textcoords="offset points", xytext=(6, 3), fontsize=9)

    ax.set_xlabel("Throughput (tokens/sec)  ↑ better", fontsize=12)
    ax.set_ylabel("Perplexity (WikiText-2)  ↓ better", fontsize=12)
    ax.set_title("Quality–Speed Pareto Frontier", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_quant_error_distribution(
    per_layer_errors: List[float],
    label: str,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Histogram of per-layer quantisation MSE."""
    plt, _ = _check_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_layer_errors, bins=30, color="#E84A5F", edgecolor="white",
            alpha=0.85, linewidth=0.5)
    ax.axvline(float(np.mean(per_layer_errors)), color="#2A363B",
               linestyle="--", linewidth=1.5, label=f"mean={np.mean(per_layer_errors):.2e}")
    ax.set_xlabel("Layer Quantisation MSE", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Per-Layer Quantisation Error — {label}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV / JSON persistence
# ---------------------------------------------------------------------------

def save_results_csv(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"  Results saved → {path}")


def save_results_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Results saved → {path}")
