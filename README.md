# **HQQ 1-bit Quantization: High-Quality Compression for Large Language Models**

## 1. Background & Motivation

Modern large language models (LLMs), such as GPT, LLaMA, or Falcon, contain **billions to trillions of parameters**.
These models achieve impressive capabilities — but at a huge cost:

* **Model size:** Hundreds of gigabytes even in 16-bit precision.
* **Compute demand:** Inference requires multiple high-end GPUs.
* **Deployment challenge:** Hard to run on consumer devices or even single GPUs.

### Traditional Solution — Quantization

Quantization compresses models by **representing weights with fewer bits**, e.g.:

* FP16 → 16 bits
* INT8 → 8 bits
* 4-bit (e.g. QLoRA, GPTQ) → 4 bits

But quantizing further — **below 4-bit** — causes **severe accuracy degradation**.
That’s where **HQQ** comes in.

---

## 2. What is HQQ?

**HQQ** stands for **High-Quality Quantization**,
a recently proposed technique that allows **extreme compression (e.g., 2-bit or even 1-bit)**
**without significant accuracy loss**.

In short:

> HQQ provides a mathematically optimized, layer-aware quantization scheme that preserves model quality even at 1-bit per weight.

This means — models like LLaMA-2 or Mistral can be stored and served **50× smaller** than their original size,
while retaining **>99% of baseline performance**.

---

## 3. Core Idea

### Traditional quantization (e.g., uniform 4-bit)

uses a fixed scaling factor:

$$
\tilde{w}_i = \text{round}\left(\frac{w_i}{s}\right) \times s
$$


where ( s ) is the quantization scale shared across many weights.

This uniform scaling causes large rounding errors —
especially when weights have diverse magnitudes (as in transformer layers).

---

### HQQ’s Key Insight:

Instead of using one fixed scale,
**HQQ dynamically adapts the quantization range per group or per tensor,**
and **learns optimal reconstruction parameters** to reduce the error.

Its optimization goal is:

$$
\min_{q, s} | W - s \cdot q |_2^2
$$

subject to ( q \in {-1, +1} )  (for 1-bit quantization).

This is solved **analytically** or via **iterative updates**,
so that each quantized tensor best approximates its original high-precision version.

In other words —

> HQQ “learns” how to quantize optimally, instead of simply rounding weights.

---

##  4. Why 1-bit is Special

A **1-bit model** stores each weight as just a binary sign:
positive (+1) or negative (-1).

So:

* Storage reduction = **16× smaller than FP16**
* Computation simplifies to **bitwise operations (XNOR + popcount)**,
  enabling ultra-fast inference on specialized hardware.

The challenge is maintaining model accuracy —
and **HQQ** tackles this by **layer-specific calibration and re-scaling** during quantization.

---

##  5. HQQ Technical Pipeline

Here’s how HQQ typically works in practice 👇

### **Step 1: Load pretrained model**

Start with a standard pretrained model (e.g., LLaMA2-7B in FP16).

### **Step 2: Analyze weight distributions**

Compute per-layer statistics to estimate optimal scaling ranges.

### **Step 3: Quantize weights**

Quantize each weight tensor ( W ) to 1-bit using optimized scales ( s_i ):

$$
\tilde{W}_i = s_i \cdot \text{sign}(W_i)
$$

### **Step 4: Calibrate**

Perform a few forward passes over a calibration dataset (e.g., 512 samples)
to minimize reconstruction error layer-by-layer.

### **Step 5: Deploy**

Save and serve the quantized model —
it loads **50× faster**, fits into a **few gigabytes**, and runs smoothly on a **single consumer GPU**.

---

## 6. Quantization Comparison

| Method       | Precision   | Memory Saving | Typical Accuracy Drop | Notes                                |
| ------------ | ----------- | ------------- | --------------------- | ------------------------------------ |
| FP16         | 16-bit      | –             | 0%                    | Baseline                             |
| INT8 (8-bit) | 8-bit       | 2×            | <1%                   | Standard for deployment              |
| GPTQ / QLoRA | 4-bit       | 4×            | 0–1%                  | Quantized fine-tuning possible       |
| AWQ          | 3–4 bit     | 4×–5×         | 0–1%                  | Activation-aware                     |
| **HQQ**      | **1–2 bit** | **8×–16×**    | **≈0–1%**             | **High-fidelity 1-bit quantization** |

---

##  7. Why HQQ Works So Well

HQQ avoids the pitfalls of older quantizers by:

1. **Per-channel scaling** — each channel has its own optimized scale.
2. **Layer-wise reconstruction loss minimization** — ensures minimal accuracy loss.
3. **Hybrid precision fallback** — critical layers (e.g., embeddings, output) can stay in 4-bit or 8-bit.
4. **Closed-form or iterative optimization** — achieves quantization error near theoretical minimum.

These improvements together make 1-bit feasible for real deployment.

---

## 8. Practical Benefits

| Aspect               | Improvement                      |
| -------------------- | -------------------------------- |
| **Storage**          | Up to 16× smaller models         |
| **Memory footprint** | 10× less GPU VRAM required       |
| **Inference speed**  | Up to 3–5× faster on GPUs        |
| **Throughput**       | Efficient bitwise matrix ops     |
| **Accuracy**         | Comparable to 4-bit quantization |

For example:

* A **70B LLaMA2** (400GB in FP16) can fit into ~25GB with HQQ 1-bit quantization.
* A **13B model** can fit into laptop GPUs while keeping BLEU / perplexity scores nearly unchanged.

---

## 9. Relationship to Other Quantization Methods

| Method                 | Type                 | Adaptation          | Notes                                  |
| ---------------------- | -------------------- | ------------------- | -------------------------------------- |
| **GPTQ**               | Post-training        | Static              | Fast, but prone to rounding error      |
| **AWQ**                | Post-training        | Activation-aware    | Better for transformer layers          |
| **BitsAndBytes 4-bit** | Runtime quantization | Simple, widely used |                                        |
| **HQQ (1-bit)**        | Learned quantization | Dynamic + optimized | Extreme compression with high fidelity |

So HQQ can be viewed as **the next generation of quantization**,
extending GPTQ/AWQ principles to ultra-low-bit representations without accuracy collapse.

---

## 10. Theoretical Insight (Simplified)

In 1-bit quantization,
we represent weight vectors ![equation](https://latex.codecogs.com/png.latex?\mathbf{w}%20\in%20\mathbb{R}^n%20\text{%20is%20quantized%20as%20}%20\mathbf{q}%20\in%20\{-1,+1\}^n)

HQQ finds the scaling s^* minimizing quantization error:


$$
s^* = \frac{\mathbf{w}^\top \mathbf{q}}{|\mathbf{q}|_2^2}
$$

Then reconstructs:

$$
\tilde{\mathbf{w}} = s^* \mathbf{q}
$$


This yields the **least-squares optimal binary approximation** of ( \mathbf{w} ),
thus achieving high fidelity even at 1-bit precision.

---


##  References

* **Frantar, M., et al. (2024)** — *HQQ: High-Quality Quantization for Large Language Models*
* Dettmers et al. (2023) — *QLoRA: Efficient Finetuning of Quantized LLMs*
* Hu et al. (2021) — *LoRA: Low-Rank Adaptation of LLMs*
* Lin et al. (2023) — *AWQ: Activation-Aware Weight Quantization for LLMs*
* Frantar & Alistarh (2022) — *Optimal Brain Compression*

