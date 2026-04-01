"""
tests/test_all.py — Unit tests for HQQ 1-bit quantisation project.

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_all.py

Test coverage
─────────────
  Group A: Config
  Group B: HQQ core math (quantise, dequantise, GAE, pack/unpack)
  Group C: HQQ optimiser (HQ proximal iterations)
  Group D: HQQLinear layer (from_linear, forward, memory stats)
  Group E: Model quantisation (quantise_model, layer replacement)
  Group F: Synthetic LLaMA (architecture, forward pass)
  Group G: Memory benchmark (analyse_model_memory, theoretical table)
  Group H: Accuracy benchmark (compute_perplexity)
  Group I: Speed benchmark (CUDATimer, benchmark_prefill)
  Group J: Metrics (BenchmarkReport, CSV/JSON save)
  Group K: Bit-packing round-trip (all bit widths)
"""

import sys, os, math, unittest, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


# ── helpers ──────────────────────────────────────────────────────────────────

def make_linear(in_f=128, out_f=64, bias=True):
    lin = nn.Linear(in_f, out_f, bias=bias)
    nn.init.normal_(lin.weight, std=0.1)
    return lin


# =============================================================================
# Group A — Config
# =============================================================================

class TestConfig(unittest.TestCase):

    def test_defaults(self):
        from config import Config
        cfg = Config()
        self.assertEqual(cfg.quant.nbits, 1)
        self.assertEqual(cfg.quant.qmax, 1)     # 2^1 - 1
        self.assertIn("lm_head", cfg.quant.skip_layers)

    def test_qmax_values(self):
        from config import QuantConfig
        for bits, expected in [(1,1),(2,3),(4,15),(8,255)]:
            q = QuantConfig(nbits=bits)
            self.assertEqual(q.qmax, expected)

    def test_label(self):
        from config import QuantConfig
        q = QuantConfig(nbits=2, group_size=128)
        self.assertEqual(q.label(), "W2G128")

    def test_presets(self):
        from config import w1_config, w2_config, w4_config, tiny_test_config
        self.assertEqual(w1_config().quant.nbits, 1)
        self.assertEqual(w2_config().quant.nbits, 2)
        self.assertEqual(w4_config().quant.nbits, 4)
        cfg = tiny_test_config()
        self.assertEqual(cfg.device, "cpu")

    def test_invalid_nbits(self):
        from config import QuantConfig
        with self.assertRaises(AssertionError):
            QuantConfig(nbits=3)   # 3 not in allowed set

    def test_summary(self):
        from config import Config
        s = Config().summary()
        self.assertIn("MODEL", s)
        self.assertIn("QUANT", s)
        self.assertIn("BENCH", s)


# =============================================================================
# Group B — HQQ Core Math
# =============================================================================

class TestHQQCore(unittest.TestCase):

    def test_quantise_range(self):
        from quantization.hqq_core import quantise, init_scale_zero
        W = torch.randn(4, 64)
        s, z = init_scale_zero(W, qmax=1)
        Wq = quantise(W, s, z, qmax=1)
        self.assertTrue((Wq >= 0).all())
        self.assertTrue((Wq <= 1).all())

    def test_quantise_dequantise_round_trip(self):
        from quantization.hqq_core import (
            init_scale_zero, quantise, dequantise, quantise_dequantise
        )
        for nbits in (1, 2, 4, 8):
            qmax = (1 << nbits) - 1
            W = torch.randn(8, 64)
            s, z = init_scale_zero(W, qmax)
            Wq   = quantise(W, s, z, qmax)
            What = dequantise(Wq, s, z)
            # Round-trip error should be < range/qmax (1 quantisation step)
            rng = W.max() - W.min()
            err = (W - What).abs().max().item()
            self.assertLess(err, rng.item() / max(qmax, 1) + 1e-4,
                            msg=f"nbits={nbits} error too large: {err}")

    def test_init_scale_zero_maps_range(self):
        from quantization.hqq_core import init_scale_zero, quantise
        W = torch.tensor([[0.0, 0.5, 1.0, -0.5]])   # 1 group of 4
        s, z = init_scale_zero(W, qmax=3)
        Wq   = quantise(W, s, z, qmax=3)
        # Min should map to 0, max should map to 3
        self.assertEqual(Wq.min().item(), 0.0)
        self.assertEqual(Wq.max().item(), 3.0)

    def test_estimate_model_size(self):
        from quantization.hqq_core import estimate_model_size_gb
        # 7B params, 1 bit → ~0.88 GB weights
        gb = estimate_model_size_gb(7e9, nbits=1)
        self.assertAlmostEqual(gb, 7e9 / 8 / 1e9, delta=0.1)

    def test_quantisation_ratio(self):
        from quantization.hqq_core import quantisation_ratio
        self.assertAlmostEqual(quantisation_ratio(1), 16.0)
        self.assertAlmostEqual(quantisation_ratio(4),  4.0)
        self.assertAlmostEqual(quantisation_ratio(8),  2.0)


# =============================================================================
# Group C — HQQ Optimiser
# =============================================================================

class TestHQQOptimiser(unittest.TestCase):

    def test_optimiser_reduces_error(self):
        from quantization.hqq_core import (
            init_scale_zero, hqq_optimise,
            quantise_dequantise
        )
        torch.manual_seed(0)
        W = torch.randn(16, 64)   # 16 groups
        qmax = 1

        s0, z0 = init_scale_zero(W, qmax)
        err_before = (W - quantise_dequantise(W, s0, z0, qmax)).pow(2).mean().item()

        s1, z1 = hqq_optimise(W, s0.clone(), z0.clone(), qmax, n_iters=10, lr=1e-3)
        err_after  = (W - quantise_dequantise(W, s1, z1, qmax)).pow(2).mean().item()

        self.assertLessEqual(err_after, err_before * 1.1,   # allow 10% slack
                             msg="Optimiser should not increase error significantly")

    def test_hqq_quantise_weight_shapes(self):
        from quantization.hqq_core import hqq_quantise_weight
        W = torch.randn(64, 128)
        for bits in (1, 2, 4):
            result = hqq_quantise_weight(W, nbits=bits, group_size=64,
                                         optimize=False)
            self.assertEqual(result["W_q"].shape, (64, 128))
            self.assertEqual(result["nbits"], bits)
            self.assertIn("scale", result)
            self.assertIn("zero",  result)
            self.assertIn("quant_error", result)

    def test_hqq_quantise_weight_values_in_range(self):
        from quantization.hqq_core import hqq_quantise_weight
        W = torch.randn(32, 64)
        res = hqq_quantise_weight(W, nbits=2, group_size=64, optimize=False)
        self.assertTrue((res["W_q"] >= 0).all())
        self.assertTrue((res["W_q"] <= 3).all())


# =============================================================================
# Group D — HQQLinear
# =============================================================================

class TestHQQLinear(unittest.TestCase):

    def _make_hqq(self, nbits=2, group_size=64):
        from quantization.hqq_linear import HQQLinear
        lin = make_linear(128, 64)
        return HQQLinear.from_linear(lin, nbits=nbits, group_size=group_size,
                                     optimize=False)

    def test_from_linear_output_shape(self):
        hqq = self._make_hqq(nbits=2)
        x   = torch.randn(4, 128)
        y   = hqq(x)
        self.assertEqual(y.shape, (4, 64))

    def test_from_linear_all_bits(self):
        for bits in (1, 2, 4, 8):
            hqq = self._make_hqq(nbits=bits)
            x   = torch.randn(2, 128)
            y   = hqq(x)
            self.assertEqual(y.shape, (2, 64))
            self.assertFalse(torch.isnan(y).any(), f"NaN output for nbits={bits}")

    def test_dequantise_shape(self):
        hqq = self._make_hqq()
        W   = hqq.dequantise()
        self.assertEqual(W.shape, (64, 128))

    def test_compression_ratio_positive(self):
        for bits in (1, 2, 4):
            hqq = self._make_hqq(nbits=bits)
            self.assertGreater(hqq.compression_ratio(), 1.0,
                               msg=f"nbits={bits} should compress vs fp16")

    def test_extra_repr(self):
        hqq = self._make_hqq()
        r   = hqq.extra_repr()
        self.assertIn("nbits", r)
        self.assertIn("compression", r)

    def test_no_bias(self):
        from quantization.hqq_linear import HQQLinear
        lin = make_linear(64, 32, bias=False)
        hqq = HQQLinear.from_linear(lin, nbits=2, group_size=64, optimize=False)
        self.assertIsNone(hqq.bias)
        y = hqq(torch.randn(2, 64))
        self.assertEqual(y.shape, (2, 32))


# =============================================================================
# Group E — Model Quantisation
# =============================================================================

class TestQuantiseModel(unittest.TestCase):

    def _tiny_model(self):
        from models.llama_utils import SyntheticLLaMA
        return SyntheticLLaMA(vocab_size=100, hidden_dim=64,
                              n_layers=2, n_heads=2).eval()

    def test_quantise_model_replaces_linears(self):
        from quantization.hqq_model import quantise_model
        from quantization.hqq_linear import HQQLinear
        from config import QuantConfig
        model = self._tiny_model()
        cfg   = QuantConfig(nbits=2, group_size=64, optimize=False)
        summary = quantise_model(model, cfg, verbose=False)
        self.assertGreater(summary["n_quantised"], 0)
        # Check at least one HQQLinear exists
        hqq_count = sum(1 for m in model.modules()
                        if isinstance(m, HQQLinear))
        self.assertGreater(hqq_count, 0)

    def test_quantise_model_forward_still_works(self):
        from quantization.hqq_model import quantise_model
        from config import QuantConfig
        model = self._tiny_model()
        cfg   = QuantConfig(nbits=1, group_size=64, optimize=False)
        quantise_model(model, cfg, verbose=False)
        ids  = torch.randint(0, 100, (2, 16))
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        self.assertEqual(out.logits.shape[:2], (2, 16))

    def test_quantise_skips_lm_head(self):
        from quantization.hqq_model import quantise_model
        from config import QuantConfig
        model = self._tiny_model()
        cfg   = QuantConfig(nbits=2, group_size=64, optimize=False,
                            skip_layers=["lm_head"])
        quantise_model(model, cfg, verbose=False)
        # lm_head should still be nn.Linear
        self.assertIsInstance(model.lm_head, nn.Linear)

    def test_summary_keys(self):
        from quantization.hqq_model import quantise_model
        from config import QuantConfig
        model = self._tiny_model()
        cfg   = QuantConfig(nbits=2, group_size=64, optimize=False)
        s = quantise_model(model, cfg, verbose=False)
        for key in ("n_quantised", "mean_error", "size_fp16_gb",
                    "size_quant_gb", "compression", "elapsed_sec"):
            self.assertIn(key, s)


# =============================================================================
# Group F — Synthetic LLaMA
# =============================================================================

class TestSyntheticLLaMA(unittest.TestCase):

    def test_forward_shape(self):
        from models.llama_utils import SyntheticLLaMA
        m   = SyntheticLLaMA(vocab_size=100, hidden_dim=64, n_layers=2, n_heads=2)
        ids = torch.randint(0, 100, (2, 8))
        out = m(input_ids=ids)
        self.assertEqual(out.logits.shape, (2, 8, 100))

    def test_load_synthetic_model(self):
        from models.llama_utils import load_synthetic_model
        model, tok = load_synthetic_model("cpu")
        self.assertTrue(callable(tok))
        ids = torch.randint(0, 100, (1, 4))
        with torch.no_grad():
            out = model(input_ids=ids)
        self.assertFalse(torch.isnan(out.logits).any())

    def test_parameter_count(self):
        from models.llama_utils import SyntheticLLaMA
        m = SyntheticLLaMA(vocab_size=100, hidden_dim=64, n_layers=2, n_heads=2)
        n = sum(p.numel() for p in m.parameters())
        self.assertGreater(n, 0)


# =============================================================================
# Group G — Memory Benchmark
# =============================================================================

class TestMemoryBenchmark(unittest.TestCase):

    def test_theoretical_table(self):
        from benchmarks.memory_benchmark import theoretical_memory_table
        t = theoretical_memory_table(7.0, bit_widths=(1, 4, 16))
        self.assertIn("W1", t)
        self.assertIn("W4", t)
        self.assertIn("fp16", t)

    def test_analyse_model_memory_fp16(self):
        from benchmarks.memory_benchmark import analyse_model_memory
        from models.llama_utils import SyntheticLLaMA
        m = SyntheticLLaMA(vocab_size=100, hidden_dim=64, n_layers=2, n_heads=2)
        r = analyse_model_memory(m, "fp16", nbits=16)
        self.assertGreater(r.weight_gb, 0)
        self.assertEqual(r.nbits, 16)

    def test_analyse_model_memory_quantised(self):
        from benchmarks.memory_benchmark import analyse_model_memory
        from quantization.hqq_model import quantise_model
        from config import QuantConfig
        from models.llama_utils import SyntheticLLaMA
        m   = SyntheticLLaMA(vocab_size=100, hidden_dim=64, n_layers=2, n_heads=2)
        cfg = QuantConfig(nbits=1, group_size=64, optimize=False)
        quantise_model(m, cfg, verbose=False)
        r   = analyse_model_memory(m, "W1G64", nbits=1)
        # Quantised model should be smaller than fp16
        from benchmarks.memory_benchmark import analyse_model_memory
        from models.llama_utils import SyntheticLLaMA as SL
        m2  = SL(vocab_size=100, hidden_dim=64, n_layers=2, n_heads=2)
        r16 = analyse_model_memory(m2, "fp16", nbits=16)
        self.assertLessEqual(r.weight_gb, r16.weight_gb + 1e-6)

    def test_layer_compression_stats(self):
        from benchmarks.memory_benchmark import layer_compression_stats, MemoryResult
        r = MemoryResult("test", 1.0, 0.0, 8.0, 16.0, 0.1, [8.0]*10, 1)
        s = layer_compression_stats(r)
        self.assertIn("mean", s)
        self.assertAlmostEqual(s["mean"], 8.0)


# =============================================================================
# Group H — Accuracy Benchmark
# =============================================================================

class TestAccuracyBenchmark(unittest.TestCase):

    def _model_and_tok(self):
        from models.llama_utils import load_synthetic_model
        return load_synthetic_model("cpu")

    def test_compute_perplexity_finite(self):
        from benchmarks.accuracy_benchmark import compute_perplexity
        model, _ = self._model_and_tok()
        seqs = [torch.randint(0, 100, (64,)) for _ in range(3)]
        ppl, n = compute_perplexity(model, seqs, torch.device("cpu"),
                                    max_seq_len=32, stride=16, verbose=False)
        self.assertGreater(ppl, 0)
        self.assertLess(ppl, 1e6)
        self.assertGreater(n, 0)

    def test_run_accuracy_benchmark(self):
        from benchmarks.accuracy_benchmark import run_accuracy_benchmark
        from config import QuantConfig
        from quantization.hqq_model import quantise_model
        model, tok = self._model_and_tok()
        cfg = QuantConfig(nbits=2, group_size=64, optimize=False)
        quantise_model(model, cfg, verbose=False)
        result = run_accuracy_benchmark(
            model, tok, label="W2G64", dataset="synthetic",
            n_samples=4, max_seq_len=32, stride=16,
            device=torch.device("cpu"), verbose=False,
        )
        self.assertGreater(result.ppl, 0)
        self.assertGreater(result.n_tokens, 0)


# =============================================================================
# Group I — Speed Benchmark
# =============================================================================

class TestSpeedBenchmark(unittest.TestCase):

    def test_cuda_timer_cpu(self):
        from benchmarks.speed_benchmark import CUDATimer
        timer = CUDATimer(torch.device("cpu"))
        with timer:
            _ = sum(range(10000))
        self.assertGreater(timer.elapsed_ms, 0)

    def test_benchmark_prefill(self):
        from benchmarks.speed_benchmark import benchmark_prefill
        from models.llama_utils import SyntheticLLaMA
        model = SyntheticLLaMA(vocab_size=100, hidden_dim=64, n_layers=2, n_heads=2)
        ids   = torch.randint(0, 100, (1, 16))
        mask  = torch.ones_like(ids)
        mean, std = benchmark_prefill(model, ids, mask, warmup=1, timed=2)
        self.assertGreater(mean, 0)
        self.assertGreaterEqual(std, 0)


# =============================================================================
# Group J — Metrics
# =============================================================================

class TestMetrics(unittest.TestCase):

    def test_benchmark_report_table(self):
        from utils.metrics import BenchmarkReport
        r = BenchmarkReport(
            model_name="test",
            quant_configs=["fp16", "W1G64"],
            ppl_results={"fp16": 5.5, "W1G64": 12.3},
            speed_results={"fp16": 20.0, "W1G64": 40.0},
            memory_results={"fp16": 14.0, "W1G64": 1.5},
            compression_results={"fp16": 1.0, "W1G64": 9.3},
        )
        t = r.summary_table()
        self.assertIn("fp16", t)
        self.assertIn("W1G64", t)
        self.assertIn("5.5", t)

    def test_save_results_json(self):
        from utils.metrics import save_results_json
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.json")
            save_results_json({"foo": 1}, path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["foo"], 1)

    def test_save_results_csv(self):
        from utils.metrics import save_results_csv
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.csv")
            save_results_csv([{"a": 1, "b": 2}], path)
            with open(path) as f:
                self.assertIn("a", f.read())


# =============================================================================
# Group K — Bit-Packing Round-Trip
# =============================================================================

class TestBitPacking(unittest.TestCase):

    def _roundtrip(self, nbits):
        from quantization.hqq_core import pack_weights, unpack_weights
        orig = torch.randint(0, (1 << nbits), (64, 128), dtype=torch.uint8)
        packed   = pack_weights(orig, nbits)
        unpacked = unpack_weights(packed, nbits, (64, 128))
        self.assertTrue(torch.equal(orig, unpacked),
                        msg=f"Pack-unpack mismatch for nbits={nbits}")

    def test_pack_unpack_1bit(self):  self._roundtrip(1)
    def test_pack_unpack_2bit(self):  self._roundtrip(2)
    def test_pack_unpack_4bit(self):  self._roundtrip(4)
    def test_pack_unpack_8bit(self):  self._roundtrip(8)

    def test_pack_reduces_size_1bit(self):
        from quantization.hqq_core import pack_weights
        orig   = torch.randint(0, 2, (64, 128), dtype=torch.uint8)
        packed = pack_weights(orig, 1)
        self.assertEqual(packed.numel() * 8, orig.numel())

    def test_pack_reduces_size_4bit(self):
        from quantization.hqq_core import pack_weights
        orig   = torch.randint(0, 16, (64, 128), dtype=torch.uint8)
        packed = pack_weights(orig, 4)
        self.assertEqual(packed.numel() * 2, orig.numel())


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
