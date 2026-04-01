"""
benchmarks/accuracy_benchmark.py — Perplexity and downstream accuracy.

Metrics
───────
  Perplexity (PPL)
      The standard measure for LM quality.
      PPL = exp(mean cross-entropy over tokens)
      Lower is better.  Base GPT-2 ≈ 30 on WikiText-2;
      LLaMA-2-7B fp16 ≈ 5.5.

  Sliding-window PPL
      For sequences longer than the model context, we use the
      Hugging Face sliding-window approach:
          • Process overlapping chunks of length max_len
          • Use a stride so only the last `stride` tokens are evaluated
          • This avoids ignoring long-range context

  Bits-per-character (BPC) / bits-per-word (BPW)
      Alternative info-theoretic measures:
          BPC = PPL_char / log(2)
          BPW = PPL_word / log(2)

  Zero-shot accuracy (optional, requires lm-eval-harness)
      Tasks: lambada_openai, arc_easy, hellaswag, winogrande
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AccuracyResult:
    label: str
    dataset: str
    ppl: float
    bpc: float          # bits per character
    n_tokens: int
    n_samples: int

    def to_dict(self) -> Dict:
        return vars(self)

    def __str__(self) -> str:
        return (
            f"[{self.label}] {self.dataset}  "
            f"PPL={self.ppl:.4f}  BPC={self.bpc:.4f}  "
            f"tokens={self.n_tokens:,}  samples={self.n_samples}"
        )


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    token_sequences: List[torch.Tensor],
    device: torch.device,
    max_seq_len: int = 2048,
    stride: int = 512,
    batch_size: int = 1,
    verbose: bool = True,
) -> Tuple[float, int]:
    """
    Compute sliding-window perplexity over a list of tokenised sequences.

    Algorithm (Hugging Face standard):
      For each sequence S of length L:
        For window [i, i+max_seq_len]:
          Evaluate NLL on tokens in [i+stride, i+max_seq_len]
          (using the full window as context)
        Accumulate NLL and token count

    Args:
        model          : the LM (fp16 or quantised)
        token_sequences: list of 1-D LongTensors
        device         : inference device
        max_seq_len    : context window length
        stride         : stride between windows (<= max_seq_len)
        batch_size     : sequences per forward pass (1 for long sequences)

    Returns:
        (perplexity, total_token_count)
    """
    model.eval()
    total_nll   = 0.0
    total_count = 0

    iterator = tqdm(token_sequences, desc="PPL eval", disable=not verbose)

    for seq in iterator:
        seq = seq.to(device)
        L   = seq.size(0)

        # Sliding window
        prev_end = 0
        for begin in range(0, L, stride):
            end    = min(begin + max_seq_len, L)
            window = seq[begin:end].unsqueeze(0)    # (1, window_len)

            target_begin = max(prev_end - begin, 0)  # tokens newly evaluated

            mask = torch.ones_like(window)
            out  = model(input_ids=window, attention_mask=mask)

            # Logits → log-probs
            logits  = out.logits[0]                  # (window_len, vocab)
            log_probs = F.log_softmax(logits, dim=-1) # (window_len, vocab)

            # Cross-entropy on the target slice (shifted by 1)
            if target_begin < window.size(1) - 1:
                tgt_tokens  = window[0, target_begin + 1 :]
                pred_logits = logits[target_begin : -1]

                nll = F.cross_entropy(
                    pred_logits, tgt_tokens, reduction="sum"
                ).item()
                total_nll   += nll
                total_count += tgt_tokens.size(0)

            prev_end = end
            if end == L:
                break

        if total_count > 0:
            iterator.set_postfix({"PPL": f"{math.exp(total_nll / total_count):.2f}"})

    if total_count == 0:
        return float("inf"), 0

    ppl = math.exp(total_nll / total_count)
    return ppl, total_count


# ---------------------------------------------------------------------------
# Dataset loading helpers (WikiText-2, C4, PTB)
# ---------------------------------------------------------------------------

def load_wikitext2(
    tokenizer,
    n_samples: int = 128,
    split: str = "test",
    max_len: int = 2048,
) -> List[torch.Tensor]:
    """Load and tokenise WikiText-2 test set."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(ds["text"])
    except Exception:
        logger.warning("Could not load WikiText-2, using synthetic text")
        text = _synthetic_text(n_samples * 200)

    return _tokenise_chunks(text, tokenizer, n_samples, max_len)


def load_c4(
    tokenizer,
    n_samples: int = 128,
    max_len: int = 2048,
) -> List[torch.Tensor]:
    """Load a slice of the C4 validation set."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for i, row in enumerate(ds):
            if i >= n_samples * 5:
                break
            texts.append(row["text"])
        text = "\n\n".join(texts)
    except Exception:
        logger.warning("Could not load C4, using synthetic text")
        text = _synthetic_text(n_samples * 200)

    return _tokenise_chunks(text, tokenizer, n_samples, max_len)


def _synthetic_text(length: int) -> str:
    """Generate plausible-ish English text for offline testing."""
    import random
    words = [
        "the", "a", "an", "of", "in", "to", "and", "for", "with",
        "model", "language", "network", "layer", "weight", "training",
        "inference", "token", "attention", "transformer", "quantisation",
        "llama", "benchmark", "accuracy", "performance", "memory",
    ]
    rng = random.Random(42)
    return " ".join(rng.choice(words) for _ in range(length))


def _tokenise_chunks(
    text: str,
    tokenizer,
    n_samples: int,
    max_len: int,
) -> List[torch.Tensor]:
    """Tokenise a long text and split into chunks of max_len."""
    try:
        ids = tokenizer.encode(text)
    except Exception:
        # Fallback: simple character-level encoding
        ids = [ord(c) % 1000 + 3 for c in text]

    ids_t  = torch.tensor(ids, dtype=torch.long)
    chunks = []
    for i in range(0, min(len(ids_t), n_samples * max_len), max_len):
        chunk = ids_t[i : i + max_len]
        if chunk.size(0) >= 32:
            chunks.append(chunk)
        if len(chunks) >= n_samples:
            break
    return chunks


# ---------------------------------------------------------------------------
# Main accuracy benchmark function
# ---------------------------------------------------------------------------

def run_accuracy_benchmark(
    model: nn.Module,
    tokenizer,
    label: str,
    dataset: str = "wikitext2",
    n_samples: int = 128,
    max_seq_len: int = 2048,
    stride: int = 512,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> AccuracyResult:
    """
    Run perplexity evaluation on the specified dataset.

    Args:
        model      : fp16 or quantised model
        tokenizer  : matching tokenizer
        label      : string name for this model (e.g. "LLaMA-2-7B W1G64")
        dataset    : "wikitext2" | "c4" | "ptb"
        n_samples  : number of text chunks to evaluate
        max_seq_len: sliding window length
        stride     : stride between windows
        device     : inference device

    Returns:
        AccuracyResult
    """
    if device is None:
        device = next(model.parameters()).device

    print(f"\n  Accuracy benchmark: {label}  dataset={dataset}")

    # Load dataset
    if dataset == "wikitext2":
        sequences = load_wikitext2(tokenizer, n_samples=n_samples, max_len=max_seq_len)
    elif dataset == "c4":
        sequences = load_c4(tokenizer, n_samples=n_samples, max_len=max_seq_len)
    else:
        logger.warning(f"Unknown dataset {dataset}, falling back to synthetic")
        sequences = [torch.randint(3, 1000, (max_seq_len,)) for _ in range(n_samples)]

    # Compute PPL
    ppl, n_toks = compute_perplexity(
        model, sequences, device,
        max_seq_len=max_seq_len,
        stride=stride,
        verbose=verbose,
    )

    bpc = math.log2(ppl) if ppl < float("inf") else float("inf")

    result = AccuracyResult(
        label=label,
        dataset=dataset,
        ppl=ppl,
        bpc=bpc,
        n_tokens=n_toks,
        n_samples=len(sequences),
    )
    print(f"  {result}")
    return result


# ---------------------------------------------------------------------------
# Quantisation accuracy table (across bit widths)
# ---------------------------------------------------------------------------

def accuracy_vs_bits_table(
    results_by_label: Dict[str, AccuracyResult],
) -> str:
    """Format a comparison table of PPL across quantisation configs."""
    header = f"{'Config':<25} {'Dataset':<12} {'PPL':>8} {'BPC':>8} {'Tokens':>10}"
    sep    = "─" * len(header)
    lines  = [sep, header, sep]
    for label, r in results_by_label.items():
        lines.append(
            f"{label:<25} {r.dataset:<12} {r.ppl:>8.4f} {r.bpc:>8.4f} {r.n_tokens:>10,}"
        )
    lines.append(sep)
    return "\n".join(lines)
