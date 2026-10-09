# ⚡ NVFP4-to-HiF4

[简体中文 / Chinese](README.md)

**Core NVFP4-to-HiF4 quantization algorithms for enabling NVFP4 model inference on Huawei Ascend NPUs.**

**4th place in the university division of the 2026 Huawei Algorithm Competition finals.** The selected release, `09281502`, includes Linear weight/activation quantization and Attention Q/K/V quantization. See the [technical report (Chinese)](docs/TECHNICAL_REPORT.md) for design details.

### Background and scope

Using an NVFP4 model on a target platform requires adapting its quantized representation while controlling the error introduced into matrix multiplication and Attention outputs. This project uses calibration statistics, equivalent transformations, and error feedback to construct HiF4 representations for these operations.

The code is a **CPU eager PyTorch reference implementation** of the calibration and online quantization stages. Deploying a complete model on Ascend also requires model loading, HiF4 packing, and integration with NPU kernels. This repository provides the quantization adaptation layer.

The repository publishes the core source, separate Chinese and English READMEs, and a technical report. Background, APIs, usage, and results are documented in each README. Competition task documents, attachments, datasets, and internal experiment tools are excluded.

### Methods

- **Hierarchical HiF4 quantization:** decode NVFP4 inputs and process groups of 64 values, searching an E6M2 root scale, two discrete scale levels, signs, and mantissas in increments of one quarter.
- **Linear:** learn two Householder reflections from calibration activations, combined with RMS-based equivalent scaling, channel reordering, and an H64 orthogonal transform. Activation Gram matrices weight the quantization error; blockwise error propagation and discrete code refinement improve the output approximation. Online activation quantization reuses frozen transforms, weight sensitivity, and low-rank refinement information.
- **Attention Q/K:** center K across the sequence and apply reciprocal Q/K scaling and a shared orthogonal transform. Full and causal output sensitivity guide a diagonal-plus-rank-3 error metric. Q selects 16 candidate refinement positions using observed calibration residual coverage; Q and K each perform two online refinement steps.
- **Attention V:** after selecting root and hierarchical scales, feed back the mean per-channel quantization residual within each 64-token window with a coefficient of **0.2**, then round the mantissas again. V stores no calibration state.

### Getting started

Install Python and PyTorch, then run the minimal example below from the repository root. It exercises all six interfaces using tiny CPU tensors; it demonstrates API usage rather than model accuracy or NPU performance.

```bash
python -m pip install torch
```

```python
import torch
import solution


def make_pair(rows, channels):
    # Unpacked NVFP4 values: -1 or +1; one scale per 16 values.
    quant = (2 * torch.randint(0, 2, (rows, channels)) - 1).float()
    scale = torch.ones(rows, channels // 16, dtype=torch.float32)
    return quant, scale


def reconstruct(params, shape):
    # Reconstruct the transformed HiF4 operand for this example.
    return (
        params["scale_factor"].float()
        * params["scale_lv2"].float()
        * params["scale_lv3"].float()
        * params["sign"].float()
        * params["mant"].float()
    ).reshape(shape)


torch.set_num_threads(1)
torch.manual_seed(0)

# Linear: calibrate weights once, then quantize activations with frozen state.
weight = make_pair(4, 64)
activation = make_pair(8, 64)
linear = solution.hif4_calibration_and_quantize_weight(*weight, [activation])
activation_params = solution.hif4_dynamic_quantize_activation(
    *activation, linear["activation_state"]
)
w_hat = reconstruct(linear["weight_params"], (4, 64))
a_hat = reconstruct(activation_params, (8, 64))
print("Linear output shape:", tuple((a_hat @ w_hat.T).shape))

# Attention: one Q head, one KV head, head dimension 64.
q, k, v = (make_pair(8, 64) for _ in range(3))
states = solution.hif4_calibration_attention(
    [{"q": q, "k": k, "v": v}], 1, 1, 64
)
q_params = solution.hif4_dynamic_quantize_q(*q, 1, 64, states["q_state"])
k_params = solution.hif4_dynamic_quantize_k(*k, 1, 64, states["k_state"])
v_params = solution.hif4_dynamic_quantize_v(*v, 1, 64, states["v_state"])
print("Q/K/V mantissa shapes:", [
    tuple(params["mant"].shape) for params in (q_params, k_params, v_params)
])
```

Expected output:

```text
Linear output shape: (8, 4)
Q/K/V mantissa shapes: [(8, 1, 8, 2, 4), (8, 1, 8, 2, 4), (8, 1, 8, 2, 4)]
```

For real workloads, calibrate using representative data, then pass the resulting frozen states unchanged to the matching online interfaces. Reconstructed Linear weights and activations are **paired transformed operands** for approximating `A @ W.T`; an individual reconstructed weight is not an elementwise approximation in the original weight coordinates. Q and K must likewise be used as a pair.

### Data formats

`quant` is a tensor of **unpacked NVFP4 numerical values**, not a packed 4-bit byte stream. Each scale applies to 16 values. Use CPU tensors; for an input with final dimension `C`:

```text
scale.shape = quant.shape[:-1] + (C // 16,)
```

The implementation multiplies values by scales and converts the decoded result to BF16 before further processing. The example uses FP32 inputs.

| Input | Layout |
|---|---|
| Linear weights | `[M, C]`, where `C` is a positive multiple of 64 |
| Linear activations | `[..., C]`, matching the weight channel count |
| Linear calibration data | `[(activation_quant, activation_scale), ...]` |
| Attention Q | `[T, Hq × D]` |
| Attention K/V | `[T, Hkv × D]` |
| Attention calibration data | `[{"q": (q_quant, q_scale), "k": (k_quant, k_scale), "v": (v_quant, v_scale)}, ...]` |

For Attention, `T > 0`, `Hq` must be a multiple of `Hkv`, and Q/K/V in a calibration sample share the same sequence length. The example uses `D = 64`; the common path uses head dimensions divisible by 64. See the source for shape checks. Treat each calibration state as an opaque, read-only object tied to its input configuration.

For input shape `[..., C]`, define `P = [..., C // 64]`. A HiF4 result is a dictionary with these fields:

| Field | Shape | dtype | Meaning |
|---|---|---|---|
| `scale_factor` | `P + [1, 1, 1]` | BF16 | Numerical value of the E6M2 root scale |
| `scale_lv2` | `P + [8, 1, 1]` | INT8 | Second-level scale: 1 or 2 |
| `scale_lv3` | `P + [8, 2, 1]` | INT8 | Third-level scale: 1 or 2 |
| `sign` | `P + [8, 2, 4]` | INT8 | −1, 0, or 1 |
| `mant` | `P + [8, 2, 4]` | BF16 | 0, 0.25, …, 1.75 |

These are numerical tensor representations, not packed NPU storage buffers.

### API

See [solution.py](solution.py) for the complete function signatures.

| Function | Purpose and return value |
|---|---|
| `hif4_calibration_and_quantize_weight` | Calibrate Linear and quantize weights; returns `weight_params` and `activation_state` |
| `hif4_dynamic_quantize_activation` | Quantize online activations using frozen state; returns a HiF4 dictionary |
| `hif4_calibration_attention` | Calibrate Attention; returns `q_state`, `k_state`, and `v_state` |
| `hif4_dynamic_quantize_q` | Quantize Q; returns a HiF4 dictionary |
| `hif4_dynamic_quantize_k` | Quantize K; returns a HiF4 dictionary |
| `hif4_dynamic_quantize_v` | Quantize V; returns a HiF4 dictionary |

### Competition results

**4th place in the university division of the 2026 Huawei Algorithm Competition finals.** The following results come from the official receipt for task `2146334`, submitted at `2026-09-28 13:07:40`. Both Linear and Attention reported **All cases passed**.

| Metric | Result |
|---|---:|
| Overall algorithm score | **25515.678320058352** |
| Linear Performance Total | 7070.00 |
| Linear Quality Total | 4647.10 |
| Attention Performance Total | 26125.00 |
| Attention Quality Total | 16548.31 |
| Full evaluation runtime | 753079 ms (approximately 753.08 s) |

These are recorded competition results, not a new local benchmark. The full evaluation runtime is not a measure of individual operator or end-to-end model inference latency.

### Version and packaging

The source matches the selected **09281502 (FinalsAttentionQVOnly)** release. Its SHA-256 is:

```text
91e0ff8fe6634e03523f6189162f83f8fcea20497bab044a1f13489012e8f8e7
```

To create a single-file submission archive from the repository root:

```bash
python -m zipfile -c solution.zip solution.py
```

The algorithm is unchanged by this documentation update. The repository contains the core implementation, Chinese and English READMEs, and the technical report; the generated ZIP does not need to be committed.
