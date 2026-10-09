# ⚡ NVFP4-to-HiF4

**简体中文** | [English](README.en.md)

面向华为昇腾系列 NPU 上的 NVFP4 模型推理，提供 **NVFP4 → HiF4 量化适配的核心算法**。

**2026 华为算法大赛决赛高校组第 4 名。** 项目包含 Linear 权重与激活量化，以及 Attention Q/K/V 量化。详细设计见[技术报告](docs/TECHNICAL_REPORT.md)。

## 项目背景

将 NVFP4 格式模型用于目标硬件时，需要在量化表示之间转换，并控制转换对矩阵乘法和 Attention 输出的影响。本项目利用校准数据的统计信息、等价变换和误差反馈，为 Linear 与 Attention 构造 HiF4 表示。

本仓库使用 **CPU eager PyTorch** 实现校准与在线量化，可作为昇腾模型推理流程中的量化适配层。

```text
NVFP4-to-HiF4/
├── README.md
├── README.en.md
├── solution.py
└── docs/
    └── TECHNICAL_REPORT.md
```

## 核心方法

- **HiF4 分层量化**：解码 NVFP4 输入后，以每 64 个元素为一组，搜索 E6M2 根尺度、两级离散尺度、符号与四分之一间隔尾数。
- **Linear**：利用校准激活统计学习两次 Householder 反射，结合 RMS 等价缩放、通道重排和 H64 正交变换；使用激活 Gram 矩阵加权权重量化误差，通过分块误差传播和离散修码改善输出。在线激活复用冻结的变换、权重敏感度与低秩修码信息。
- **Attention Q/K**：K 按序列去均值，Q/K 采用互逆缩放和共享正交变换；结合 Full/causal 输出敏感度，拟合对角项加 rank-3 的误差度量。Q 的 16 个修码候选位置按校准时的实际残差覆盖选择，Q/K 在线各执行两轮离散修码。
- **Attention V**：确定根尺度和层级尺度后，对每 64 token 窗口的通道平均量化残差使用 **0.2** 系数反馈，再重新舍入尾数。V 在线量化直接使用输入张量。

<a id="quick-start"></a>

## 快速入门

依赖 Python 和 PyTorch，代码另使用标准库 `math`。在已安装 Python 的环境中：

```bash
python -m pip install torch
```

在仓库根目录运行下面的示例，使用 CPU 张量依次调用六个接口。

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

预期输出：

```text
Linear output shape: (8, 4)
Q/K/V mantissa shapes: [(8, 1, 8, 2, 4), (8, 1, 8, 2, 4), (8, 1, 8, 2, 4)]
```

实际使用时，先用代表性校准数据生成状态，再将对应状态原样传给在线接口。示例中的 Linear 权重和激活是**配套变换后的操作数**，一起用于近似 `A @ W.T`。Q/K 同样配套使用。

## 输入与输出格式

`quant` 表示**已经解包为数值的 NVFP4 张量**，`scale` 为每 16 个数共享的尺度。输入张量放在 CPU 上。对最后一维为 `C` 的输入：

```text
scale.shape = quant.shape[:-1] + (C // 16,)
```

代码将数值与尺度相乘后转为 BF16，再进行后续处理。最小示例使用 FP32 输入张量。

| 场景 | 输入结构 |
|---|---|
| Linear 权重 | `[M, C]`，`C` 为正的 64 倍数 |
| Linear 激活 | `[..., C]`，通道数与权重一致 |
| Linear 校准列表 | `[(activation_quant, activation_scale), ...]` |
| Attention Q | `[T, Hq × D]` |
| Attention K/V | `[T, Hkv × D]` |
| Attention 校准列表 | `[{"q": (q_quant, q_scale), "k": (k_quant, k_scale), "v": (v_quant, v_scale)}, ...]` |

Attention 中 `T > 0`，`Hq` 为 `Hkv` 的整数倍，同一校准样本的 Q/K/V 序列长度一致；示例采用 `D = 64`。常用路径的 `D` 为 64 的倍数，具体形状检查见源码。校准状态与相应输入配置绑定，在线调用时传入对应接口。

对于输入 `[..., C]`，令前缀 `P = [..., C // 64]`，返回的 HiF4 字典包含：

| 字段 | 形状 | dtype | 含义 |
|---|---|---|---|
| `scale_factor` | `P + [1, 1, 1]` | BF16 | E6M2 根尺度的数值表示 |
| `scale_lv2` | `P + [8, 1, 1]` | INT8 | 第二级尺度，1 或 2 |
| `scale_lv3` | `P + [8, 2, 1]` | INT8 | 第三级尺度，1 或 2 |
| `sign` | `P + [8, 2, 4]` | INT8 | −1、0 或 1 |
| `mant` | `P + [8, 2, 4]` | BF16 | 0、0.25、…、1.75 |

## 六个接口

完整函数定义见 [solution.py](solution.py)。

| 接口 | 作用与返回值 |
|---|---|
| `hif4_calibration_and_quantize_weight` | Linear 校准及权重量化，返回 `weight_params` 与 `activation_state` |
| `hif4_dynamic_quantize_activation` | 使用冻结状态量化在线激活，返回 HiF4 字典 |
| `hif4_calibration_attention` | Attention 校准，返回 `q_state`、`k_state`、`v_state` |
| `hif4_dynamic_quantize_q` | 在线 Q 量化，返回 HiF4 字典 |
| `hif4_dynamic_quantize_k` | 在线 K 量化，返回 HiF4 字典 |
| `hif4_dynamic_quantize_v` | 在线 V 量化，返回 HiF4 字典 |

## 比赛成绩

**2026 华为算法大赛决赛高校组第 4 名。** 以下数据来自任务 `2146334` 的官方评测回执，提交时间为 `2026-09-28 13:07:40`，Linear 与 Attention 均为 **All cases passed**。

| 指标 | 成绩 |
|---|---:|
| 算法总分 | **25515.678320058352** |
| Linear Performance Total | 7070.00 |
| Linear Quality Total | 4647.10 |
| Attention Performance Total | 26125.00 |
| Attention Quality Total | 16548.31 |
| 整场执行用时 | 753079 ms（约 753.08 秒） |
