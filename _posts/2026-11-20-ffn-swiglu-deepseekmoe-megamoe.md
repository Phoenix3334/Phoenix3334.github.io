---
layout: post
title: "从 FFN、SwiGLU 到 DeepSeekMoE：专家怎么选，GPU 又怎样执行？"
date: 2026-11-20 15:30:00 +0800
lang: zh-CN
summary: "从 Transformer Block 与 SwiGLU 原理出发，手写 Top-k MoE，解释 DeepSeekMoE 的细粒度和共享专家、路由负载均衡，再连接 SGLang EP 与 MegaMoE 的通信计算重叠。"
tags: [Transformer, FFN, SwiGLU, MoE, DeepSeekMoE, EP, MegaMoE, SGLang]
---

之前写 [Attention 演进](/blog/2026/11/19/attention-from-gqa-to-kimi-k31/) 时，我一直在追问 KV Cache 如何存储、读取和更新。但 Transformer Block 并不只有 Attention：拿常见的 Decoder-only、Pre-Norm 架构来说，另一半就是 **FFN（Feed-Forward Network）**。我们讨论 DeepSeekMoE、Expert Parallelism、MegaMoE 时，主要是在改这一半。

最近读的 CS336 Transformer 学习材料，从 Linear、SwiGLU 一步步实现了 Dense Transformer Block。它很适合用来建立原理，但**附件本身没有实现 MLA 或 MoE**。所以下面先沿用它的 FFN 数学结构，再结合 DeepSeekMoE、DeepSeek-V3 的论文和 DeepGEMM/SGLang 公开实现解释专家与执行后端。

**先说结论：** **Attention** 负责让 Token 读取上下文；**SwiGLU FFN** 用 Gate、Up、Down 三组矩阵分别变换每个 Token；**MoE** 将一个 Dense FFN 换成多个 Expert，Router 只为每个 Token 选择少数专家；**DeepSeekMoE** 用细粒度 Routed Expert 和始终激活的 Shared Expert 改进专家分工；**EP** 决定专家存在哪张 GPU、Token 怎样跨 Rank 发送；**MegaMoE** 则把跨卡 Dispatch、专家 GEMM、激活和 Combine 协同组织，尽量重叠 NVLink 通信与 Tensor Core 计算。**MoE 是模型结构，EP 是分布式并行策略，MegaMoE 是 Kernel 执行后端。**

<p align="center">
  <a href="/assets/posts/ffn-moe-execution-layers.svg" target="_blank" rel="noopener noreferrer" title="查看 FFN MoE EP MegaMoE 总览">
    <img src="/assets/posts/ffn-moe-execution-layers.svg" width="100%" alt="从单组SwiGLU FFN到Top-k MoE，再到分布式EP和MegaMoE融合执行的三个层次">
  </a>
</p>

*图 1 · 从数学模型到分布式执行，不能把三层优化当成同一件事。*

## 1. 先把 FFN 放回 Transformer Block

一个 Pre-Norm Decoder Block 可以抽象成：

~~~python
# x: [B, S, D]
x = x + attention(norm_1(x))
x = x + ffn(norm_2(x))
# 形状仍然是 [B, S, D]
~~~

Attention 的 Query 读取其他位置的 Key/Value，因而具有跨 Token 信息交互；FFN 则对每个 Token 的向量**独立应用同一组权重**，不会再额外扫描历史 Token。虽然其输入可以是 [B, S, D]，但 FFN 的大矩阵乘法常把前两维合并成 N=B×S 个 Token，因此实质是 [N, D] 经投影到 [N, D_ff] 再回到 [N, D]。

MLA 优化的主要是 **Attention 的历史 KV 表示与读取**；FFN/MoE 关注的则是 **Token 当前经过哪些权重、多少矩阵计算及如何分布式执行**。所以 MoE 不是另一种 MLA，EP 也不是 DCP 的别名。

## 2. Dense FFN 为什么演化成 SwiGLU？

基本 FFN 是两个线性层，中间有激活函数：

~~~text
FFN(x) = W_down · activation(W_up · x)
        D → D_ff → D
~~~

没有激活函数时，两个线性映射仍然可以合成为一层；激活函数提供非线性表达能力。

SwiGLU 引入 Gate 分支，常见写法为：

~~~text
gate   = SiLU(W_gate · x)
up     = W_up · x
hidden = gate ⊙ up
y      = W_down · hidden
~~~

这里 ⊙ 表示逐元素乘法。Gate 不一定是 0/1 开关：SiLU 可以产生连续的正负数、放大或缩小特征。注意不带 Bias 的 SwiGLU 参数量为：

~~~text
W_gate: D × D_ff
W_up:   D × D_ff
W_down: D_ff × D

Total = 3 × D × D_ff
~~~

因此处理 N 个 Token 的三次 Linear，矩阵乘法的 FLOPs 量级约为 **6 × N × D × D_ff**（一次乘加按 2 FLOPs），之外还有 SiLU 与逐元素乘法。要看具体工作点：短序列或大 Token Batch 时 FFN GEMM 往往很重要；长上下文 Prefill 时 Attention 的二次项也可能占主要计算量。不能把某份 CS336 样例的 FFN FLOPs 占比当成所有模型的固定常数。

可以用最小 PyTorch 代码把三路投影看清楚：

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

x = torch.randn(2, 16, 128)
assert SwiGLU(128, 256)(x).shape == x.shape
~~~

这段代码保持了 CS336 讲解中 W1/Gate、W3/Up、W2/Down 的数学关系，使用 PyTorch 现成 Linear 只为了减少样板代码；如果要真正按 CS336 Assignment 1 手搓，还需把 Linear 的参数创建和矩阵乘法也自己实现。

## 3. MoE 不是让所有 Token 运行所有 FFN

假如 Dense 模型只有一组 SwiGLU，每个 Token 都使用这套权重。MoE 则包含多组不同参数的 Expert。对典型 Top-k Routed MoE，Router 先产生 Token 对各专家的分数，只选择 k 个，再加权合并它们的结果：

~~~text
scores = Router(x_t)
expert_ids = TopK(scores, k)

routed_output = Σ gate_i,t × Expert_i(x_t)
                i ∈ expert_ids
~~~

Top-k 选择与权重归一化在不同模型里可能不同；不是所有 MoE 都用同一种 Softmax 或路由方式。

<p align="center">
  <a href="/assets/posts/ffn-moe-token-routing.svg" target="_blank" rel="noopener noreferrer" title="查看 Token 的 Top-k Expert 路由示意">
    <img src="/assets/posts/ffn-moe-token-routing.svg" width="100%" alt="Router为Token选择Top-k专家，按专家Rank分发，分别执行SwiGLU，再按权重Combine回原Token顺序">
  </a>
</p>

*图 2 · 图中 4 个 Expert、Top-2 只是教学示例，不是特定 DeepSeek 模型配置。*

MoE 的核心优势是**总参数量可大幅增长，每个 Token 只激活一部分专家权重和计算**。但未激活专家的权重仍要放在某些设备的显存或存储层中，所以“激活参数小”不意味着“总权重无需占显存”。

下面手写一个便于验证 shape 的小型 Routed + Shared MoE：

~~~python
class SimpleMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(num_experts)
        ])
        self.shared = SwiGLU(d_model, d_ff)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        scores = self.router(flat).softmax(dim=-1)
        weights, indices = scores.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(indices == expert_id)
            if token_ids.numel() == 0:
                continue
            values = expert(flat[token_ids])
            out.index_add_(
                0, token_ids,
                values * weights[token_ids, slots].unsqueeze(-1)
            )
        return (out + self.shared(flat)).reshape(shape)

x = torch.randn(2, 8, 128)
moe = SimpleMoE(128, 256, num_experts=4, top_k=2)
assert moe(x).shape == x.shape
~~~

这段代码表达 Router → Token-to-Expert Dispatch → Expert FFN → Combine 的**数学语义**；实际生产系统不会这样逐 Expert 循环 Python 和使用大量动态索引，而是使用 Packing、Grouped GEMM、All-to-All 及融合执行。这里的简化权重归一化和 Shared Expert 也**不是对 DeepSeek-V3 的精确复刻**。

## 4. DeepSeekMoE 的两个关键设计

[DeepSeekMoE（ACL 2024）](https://aclanthology.org/2024.acl-long.70/)强调的是两项结构优化。

**Fine-Grained Expert Segmentation（细粒度专家）**：把原先较大的专家切分为更多、更小的专家，配合每 Token 激活更多个小专家。可以用 N→mN、K→mK 的概念理解：在保持总/激活参数预算大致可比的前提下，允许更多样的专家组合，让 Routed Expert 更有机会形成专长。这不是单纯地“专家数翻倍，计算也翻倍”。

**Shared Expert Isolation（共享专家隔离）**：有些通用知识没必要由所有 Routed Expert 重复学习，因此安排一部分始终激活的 Shared Expert；其他由 Router 挑选的 Routed Expert 则负责更有区别的特征。常见结构可以记成：

~~~text
x
├── Shared Expert（固定参与）
│      └── shared_output
└── Router → Routed Top-k Experts
       └── weighted_combine → routed_output

y = shared_output + routed_output
~~~

Shared Expert 会消耗每 Token 的计算预算，并不是免费的。它与 Routed Expert 的参数分布、具体个数取决于模型版本。

### 为什么 Expert 负载均衡会成为推理系统问题？

Token 路由很容易不均匀：有些 Expert 吸收大量 Token，另一些 Expert 几乎没工作。即使同一张卡的 GEMM 很快，**跨 Rank 的 Expert Skew**也会拉长同步等待，降低实际可服务吞吐。

[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)引入 **Auxiliary-Loss-Free Load Balancing**：为每个专家的 Affinity Score 加动态 Bias 来影响 **Top-k 选择**，但计算实际路由权重时仍使用未加 Bias 的原始 Affinity。这样可以调节专家选择，而不直接把负载修正量混进模型输出权重。

另外要区分：训练期的 Routing Bias、推理期单窗口 Expert Skew，与系统层的 **EPLB（Expert Parallel Load Balancing）** 并不是一回事。EPLB 可以涉及 Expert Replica 或重排的运行时策略，需要按具体版本分析。

## 5. MoE、EP、MegaMoE 是三个不同层次

| 层次 | 决定什么 | 关键组成 |
| --- | --- | --- |
| **MoE** | 哪些 Expert 被激活，输出怎样组合 | Router、Shared/Routed、Top-k |
| **EP** | Expert 存在哪个 GPU Rank，Token 如何交换 | Dispatch、All-to-All、Combine |
| **执行后端** | GEMM、通信、同步能否融合和重叠 | Grouped GEMM、MegaMoE |

当多个 Expert 分布在不同 GPU 上时，执行通常会经历：

~~~text
Token hidden states
  → Router / Top-k Expert IDs
  → Pack / Dispatch
  → EP All-to-All
  → 本地 Expert GEMM 1 → SwiGLU → GEMM 2
  → EP All-to-All / Combine
  → 恢复原 Token 顺序
~~~

这条路径里最容易被忽略的开销是动态 Token 分布、无效 Padding、大量小 GEMM、网络交换和 Rank 等待。**EP 可以减少单卡需要持有的专家权重，但通信并不免费。**

[DeepGEMM 的 MegaMoE](https://github.com/deepseek-ai/DeepGEMM)进一步将 Dispatch、Linear1、SwiGLU、Linear2 和 Combine 协同组织到融合的 GPU 执行路径，通过相应硬件能力让 **NVLink 通信与 Tensor Core 计算重叠**。它不是 MoE 的另一种数学公式，也不是简单多开两个 CUDA Stream。能隐藏多少通信，仍受网络带宽、Expert Skew、Kernel 资源占用和工作负载限制。

对比 H200 与 B300 时要格外留意：[SGLang DeepSeek-V4 Recipe](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx) 区分了 Blackwell 的 W4A8/W4A4 路径与 Hopper 的独立全 FP8 实验路径。不能仅凭功能名称就假设不同 GPU 运行相同 Kernel 或得到相同收益。

之前的 [《H200/B300、昇腾 910B、鲲鹏 920：数据搬运如何协同？》](/blog/2026/11/20/gpu-npu-cpu-inference-systems/) 从 TMA、MTE、NVLink 等硬件路径解释 MegaMoE；这一篇主要从 **FFN/Router/Expert 的模型层级**解释它。两者可以互相对照。

## 6. 回到 SGLang：局部 Kernel 快不等于 E2E 吞吐高

MoE 的运行时至少涉及三个需要区分的性能问题。

首先是 **计算问题**：Expert GEMM 的 Shape、Batch 与量化精度决定 GEMM 的效率。其次是 **通信问题**：All-to-All、Token Packing、网络和 Expert 数据布局是否位于关键路径。最后是 **调度问题**：Router 的 Expert 分布是否引起 Rank 不均衡，是否使实际有效 Batch 小于预期。

我在 [DCP A2A Send Packing PR #49455](https://github.com/sgl-project/sglang/pull/49455) 中将过多的细粒度 Program 改为按 Peer Block 聚合。在 B300 的 nsys 结果里，目标 Kernel 单层从 **15.0 μs 降到 6.9 μs**，相关每步累计耗时由 **0.361 ms 降到 0.164 ms**，但端到端吞吐仍在噪声内。它不是 MegaMoE 的实现，而是说明同一条系统规律：**Kernel 加速倍数不能直接等同于整个服务加速倍数**。

在 MoE 场景里，要判断 MegaMoE、EPLB 和传统 A2A 的收益，应当固定模型、量化、TP/EP/DP、Batch、请求长度及 SLO，再检查 nsys 的通信时间是否被真实重叠、Expert 是否均衡，最后看端到端的 Output TPS、TPOT 与稳定性。

## 7. 回到整条 Transformer Forward

~~~text
Transformer Block
  ├── Norm → Attention：MHA / GQA / MLA / DSA / KDA
  │                  └→ KV Cache / Recurrent State
  │
  └── Norm → FFN：Dense SwiGLU / Routed MoE
                    ├→ Router / Shared Expert
                    ├→ Top-k Routed Expert
                    └→ EP Dispatch → GEMM → Combine
                           └→ MegaMoE 等执行后端

整个系统还要由 Scheduler、显存管理、通信与并行拓扑保证正确交付。
~~~

**Attention 决定怎么利用上下文，FFN/MoE 决定当前 Token 如何经过模型权重，而推理 Runtime 决定这些计算是否能高效、正确地跑在硬件上。** 理解这一条链，才能把模型架构变化和 SGLang 的 EP、DCP、MegaMoE 以及端到端压测放进同一个系统视角。

---

## 参考资料

- [Stanford CS336 · Language Modeling from Scratch（Assignment 1）](https://cs336.stanford.edu/)
- [SwiGLU · GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [DeepSeekMoE · Fine-Grained Segmentation / Shared Expert（ACL 2024）](https://aclanthology.org/2024.acl-long.70/)
- [DeepSeek-V3 · Technical Report / Load Balancing](https://arxiv.org/abs/2412.19437)
- [DeepGEMM · MegaMoE](https://github.com/deepseek-ai/DeepGEMM)
- [SGLang · DeepSeek V4 推理 Recipe](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx)
- [Attention 演进：MHA/MLA/KDA/DSA](/blog/2026/11/19/attention-from-gqa-to-kimi-k31/)
- [GPU/NPU/CPU 数据搬运与 MegaMoE](/blog/2026/11/20/gpu-npu-cpu-inference-systems/)
