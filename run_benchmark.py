"""
run_benchmark.py — Full speed × accuracy × memory benchmark suite.

Runs the complete evaluation pipeline:
  1. Load fp16 baseline model
  2. For each quantisation config (W1, W2, W4, W8):
       a. Quantise model (in-place copy)
       b. Measure perplexity (WikiText-2)
       c. Measure inference speed (tokens/sec)
       d. Measure memory footprint (GB)
  3. Print comparison table
  4. Save CSV / JSON results
  5. Generate plots (PPL vs bits, speed vs bits, Pareto)

Usage
─────
  # Full benchmark with real LLaMA 2
  python run_benchmark.py --model meta-llama/Llama-2-7b-hf

  # Quick offline test with synthetic model
  python run_benchmark.py --synthetic --ppl-samples 4 --speed-runs 2

  # Only accuracy benchmark (no speed)
  python run_benchmark.py --synthetic --no-speed

  # Only speed benchmark (no PPL)
  python run_benchmark.py --synthetic --no-accuracy
"""

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="HQQ quantisation benchmark suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic model (offline, no download)")
    p.add_argument("--bits", nargs="+", type=int, default=[1, 2, 4, 8],
                   help="Bit-widths to benchmark")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--opt-iters", type=int, default=20)
    p.add_argument("--no-optimize", action="store_true")
    p.add_argument("--ppl-samples", type=int, default=128)
    p.add_argument("--speed-runs", type=int, default=5)
    p.add_argument("--speed-warmup", type=int, default=2)
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    p.add_argument("--seq-lengths", nargs="+", type=int, default=[128])
    p.add_argument("--no-accuracy", action="store_true")
    p.add_argument("--no-speed", action="store_true")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--output-dir", default="outputs/hqq/benchmark")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--dataset", default="wikitext2")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label(bits: int, group_size: int) -> str:
    return "fp16" if bits == 16 else f"W{bits}G{group_size}"


def _deep_copy_model(model):
    """Deep-copy the model for quantisation without modifying original."""
    return copy.deepcopy(model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load base model ──────────────────────────────────────────────────
    if args.synthetic:
        from models.llama_utils import load_synthetic_model
        base_model, tokenizer = load_synthetic_model(str(device))
        model_tag = "SyntheticLLaMA"
        logger.info("Using synthetic model (offline mode)")
    else:
        from models.llama_utils import load_llama
        base_model, tokenizer = load_llama(
            args.model, device_map="auto" if device.type == "cuda" else None,
            hf_token=args.hf_token,
        )
        model_tag = args.model.split("/")[-1]

    # ── Print theoretical memory table ───────────────────────────────────
    from benchmarks.memory_benchmark import theoretical_memory_table
    n_params = sum(p.numel() for p in base_model.parameters())
    print(theoretical_memory_table(n_params / 1e9, bit_widths=tuple(args.bits + [16])))

    # ── Benchmark containers ─────────────────────────────────────────────
    ppl_results:   Dict[str, float] = {}
    speed_results: Dict[str, float] = {}
    memory_results: Dict[str, float] = {}
    compression_results: Dict[str, float] = {}
    all_speed_rows: List[Dict] = []
    quant_error_per_config: Dict[str, List[float]] = {}

    all_labels = ["fp16"] + [_label(b, args.group_size) for b in args.bits]

    # ── fp16 baseline ────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print(f"  Baseline: fp16")
    print(f"{'═'*64}")

    # Accuracy
    if not args.no_accuracy:
        from benchmarks.accuracy_benchmark import run_accuracy_benchmark
        acc = run_accuracy_benchmark(
            base_model, tokenizer, label="fp16",
            dataset=args.dataset, n_samples=args.ppl_samples,
            device=device, verbose=True,
        )
        ppl_results["fp16"] = acc.ppl

    # Speed
    if not args.no_speed:
        from benchmarks.speed_benchmark import run_speed_benchmark
        speed_rows = run_speed_benchmark(
            base_model, tokenizer, label="fp16",
            batch_sizes=args.batch_sizes, seq_lengths=args.seq_lengths,
            gen_tokens=args.gen_tokens,
            warmup=args.speed_warmup, timed=args.speed_runs,
            device=device,
        )
        if speed_rows:
            speed_results["fp16"] = speed_rows[0].throughput_tok_s
            all_speed_rows.extend(r.to_dict() for r in speed_rows)

    # Memory
    if not args.no_memory:
        from benchmarks.memory_benchmark import analyse_model_memory
        mem = analyse_model_memory(base_model, "fp16", nbits=16, device=device)
        memory_results["fp16"]     = mem.weight_gb
        compression_results["fp16"] = 1.0
        print(f"  {mem}")

    # ── Quantisation sweep ───────────────────────────────────────────────
    from config import QuantConfig
    from quantization.hqq_model import quantise_model

    for bits in args.bits:
        label = _label(bits, args.group_size)
        print(f"\n{'═'*64}")
        print(f"  Quantising: {label}  (bits={bits}, group={args.group_size})")
        print(f"{'═'*64}")

        # Deep copy so we can quantise independently each time
        model_copy = _deep_copy_model(base_model).to(device)
        model_copy.eval()

        quant_cfg = QuantConfig(
            nbits=bits,
            group_size=args.group_size,
            optimize=not args.no_optimize,
            opt_iters=args.opt_iters,
        )

        t0 = time.time()
        summary = quantise_model(model_copy, quant_cfg, verbose=True)
        logger.info(f"  Quantised in {time.time()-t0:.1f}s")

        quant_error_per_config[label] = [
            s["quant_error"] for s in summary["per_layer"]
        ]

        # Accuracy
        if not args.no_accuracy:
            from benchmarks.accuracy_benchmark import run_accuracy_benchmark
            acc = run_accuracy_benchmark(
                model_copy, tokenizer, label=label,
                dataset=args.dataset, n_samples=args.ppl_samples,
                device=device, verbose=True,
            )
            ppl_results[label] = acc.ppl

        # Speed
        if not args.no_speed:
            from benchmarks.speed_benchmark import run_speed_benchmark
            speed_rows = run_speed_benchmark(
                model_copy, tokenizer, label=label,
                batch_sizes=args.batch_sizes, seq_lengths=args.seq_lengths,
                gen_tokens=args.gen_tokens,
                warmup=args.speed_warmup, timed=args.speed_runs,
                device=device,
            )
            if speed_rows:
                speed_results[label] = speed_rows[0].throughput_tok_s
                all_speed_rows.extend(r.to_dict() for r in speed_rows)

        # Memory
        if not args.no_memory:
            from benchmarks.memory_benchmark import analyse_model_memory
            mem = analyse_model_memory(model_copy, label, nbits=bits, device=device)
            memory_results[label]      = mem.weight_gb
            compression_results[label] = mem.compression_vs_fp16
            print(f"  {mem}")

        del model_copy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Results table ────────────────────────────────────────────────────
    from utils.metrics import BenchmarkReport, save_results_csv, save_results_json
    report = BenchmarkReport(
        model_name=model_tag,
        quant_configs=all_labels,
        ppl_results=ppl_results,
        speed_results=speed_results,
        memory_results=memory_results,
        compression_results=compression_results,
    )
    print(f"\n{'═'*64}")
    print(f"  BENCHMARK RESULTS — {model_tag}")
    print(f"{'═'*64}")
    print(report.summary_table())

    # ── Save results ─────────────────────────────────────────────────────
    save_results_json(report.to_dict(), str(out_dir / "results.json"))

    if all_speed_rows:
        save_results_csv(all_speed_rows, str(out_dir / "speed_results.csv"))

    ppl_csv = [{"config": k, "ppl": v} for k, v in ppl_results.items()]
    if ppl_csv:
        save_results_csv(ppl_csv, str(out_dir / "ppl_results.csv"))

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        try:
            from utils.metrics import (
                plot_ppl_vs_bits, plot_speed_vs_bits,
                plot_memory_vs_bits, plot_pareto,
                plot_quant_error_distribution,
            )
            plot_bits  = [16] + args.bits
            plot_ppls  = [ppl_results.get(_label(b, args.group_size)
                          if b != 16 else "fp16", float("nan"))
                          for b in plot_bits]
            plot_spds  = [speed_results.get(_label(b, args.group_size)
                          if b != 16 else "fp16", 0.0)
                          for b in plot_bits]
            plot_mems  = [memory_results.get(_label(b, args.group_size)
                          if b != 16 else "fp16", 0.0)
                          for b in plot_bits]

            if any(not isinstance(v, float) or not (v != v) for v in plot_ppls):
                plot_ppl_vs_bits(
                    plot_bits, plot_ppls, label=model_tag,
                    save_path=str(out_dir / "ppl_vs_bits.png"), show=False,
                )
            if any(v > 0 for v in plot_spds):
                plot_speed_vs_bits(
                    plot_bits, plot_spds, label=model_tag,
                    save_path=str(out_dir / "speed_vs_bits.png"), show=False,
                )
            if any(v > 0 for v in plot_mems):
                plot_memory_vs_bits(
                    plot_bits, plot_mems, label=model_tag,
                    save_path=str(out_dir / "memory_vs_bits.png"), show=False,
                )
            # Pareto
            if ppl_results and speed_results:
                cfgs = list(set(ppl_results) & set(speed_results))
                plot_pareto(
                    cfgs,
                    [ppl_results[c]   for c in cfgs],
                    [speed_results[c] for c in cfgs],
                    save_path=str(out_dir / "pareto.png"), show=False,
                )
            # Quant error histograms
            for lbl, errs in quant_error_per_config.items():
                if errs:
                    plot_quant_error_distribution(
                        errs, label=lbl,
                        save_path=str(out_dir / f"quant_error_{lbl}.png"),
                        show=False,
                    )
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")

    print(f"\n  ✅  All results saved to: {out_dir}")


if __name__ == "__main__":
    main()
