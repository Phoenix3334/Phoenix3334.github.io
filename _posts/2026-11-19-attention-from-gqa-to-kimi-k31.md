---
layout: post
title: "注意力机制到底在优化什么？从 MHA、MLA 到 KDA、DSA"
date: 2026-11-19 22:15:00 +0800
lang: zh-CN
summary: "沿着缓存表示、历史读取和递推状态三个维度理解 MHA/MQA/GQA、MLA、DSA、Linear/KDA；再用 Kimi K3/K3.1 说明 KV Cache 与 Recurrent State、DCP、P/D 分离和推理容量。"
tags: [Attention, GQA, MLA, DSA, KDA, Kimi-K3, Inference-Systems]
---

最近从手写 MHA、GQA 一路看到 MLA、Linear Attention、KDA 和 DSA，最困惑的地方并不是某一个公式，而是**它们似乎都在优化长上下文，却又能同时出现在同一个模型里**。例如 Kimi K3 有 69 层 KDA 和 24 层 MLA；K3.1 又在保留 KDA 的同时把那 24 层变成带 Indexer 的 DSA。MLA、KDA 和 DSA 到底是什么关系？

我后来发现，问题出在分类方式：**有的方案决定一份历史要存成什么形状，有的决定这次 Query 访问哪些历史位置，还有的直接改变历史信息的更新方式**。把它们强行画成一条从 MHA 进化到 KDA 的链，就会越来越乱。

这篇文章想沿着一个 Decode 请求往下看：历史信息怎么存、怎么读，最后又如何被 SGLang 一类推理引擎调度和分配显存。我们之前讨论过的 KV Cache、TP/DCP、Prefix Cache、Speculative Decode 和 Kimi K3/K3.1，也能放回同一张图里。

<p align="center">
  <img src="/assets/posts/attention-three-axes.svg" width="100%" alt="注意力机制的三个维度：KV表示、历史读取范围、聚合与状态更新；K3与K3.1为混合架构">
</p>

*图 1 · 先分清维度，再讨论模型。MHA/GQA/MQA、MLA、DSA 和 KDA 并非一条互斥的进化树。*

## 1. 从一次 Decode 开始：为什么会有 KV Cache？

传统 Self-Attention 可以写成：

~~~text
Q = X · Wq,   K = X · Wk,   V = X · Wv
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_head)) · V
~~~

Prefill 处理输入序列，之后 Decode 逐步生成 Token。对于第 t 个 Token，它的 Query 需要使用前面历史的 K/V。如果每步都重算所有旧 Token 的投影，代价太大，于是引擎缓存已经算出的 K/V，只在本步追加新的 K/V。

~~~text
history KV:  (K₀,V₀), (K₁,V₁), ... , (Kₜ₋₁,Vₜ₋₁)
new Qₜ ────────────────读取历史─────────────────→ Attention
new KV:                                      (Kₜ,Vₜ) → append
~~~

这里马上就出现两个不同的问题：

**容量问题：**上下文 L 越长，长期保存的 KV 越多；并发 C 越高，通常还要乘上更多请求。

**访存问题：**若使用 Full Attention，新 Query 每步都要读越来越长的历史 KV；吞吐和 TPOT 可能受显存带宽、Kernel、并行通信等制约。

所以以下机制不是都在“省 Attention 计算”：有些缩小缓存，有些只减少当前步要读取的历史，还有些改变状态表示本身。

## 2. MHA、MQA、GQA：保留多少个 KV Head？

先假设一个注意力层有 8 个 Q Head。MHA 给每个 Q Head 独立 K/V；MQA 让所有 Q Head 共享一份 K/V；GQA 让几个连续 Q Head 共享一组 K/V。Q Head 数不必因为共享 KV 而减少。

<p align="center">
  <img src="/assets/posts/attention-mha-gqa-mqa-heads.svg" width="100%" alt="同样八个Q Head，MHA保存8组KV，GQA示例保存2组KV，MQA保存1组KV">
</p>

*图 2 · 改变的是缓存中真正保存的 KV Head 数量。*

设 Hq 为 Q Head 数、Hkv 为 KV Head 数、dh 为每个 Head 维度，单层每个历史 Token 的缓存量近似为：

~~~text
KV bytes / token / layer = 2 × Hkv × dh × bytes_per_element

MHA : Hkv = Hq
GQA : 1 < Hkv < Hq
MQA : Hkv = 1
~~~

例如 Hq=8、dh=128、BF16，每层每 Token 占用从 MHA 的 4096 bytes，变成 GQA（2 KV Head）的 1024 bytes，再变成 MQA 的 512 bytes。

之前我们手写 GQA 时，最容易卡在 repeat_kv：明明计算时把 K/V 从 2 组扩展成 8 组，为什么还省显存？答案是**缓存更新在前，计算时的 Head 映射在后**。缓存只保存 2 组 KV；Eager 实现可以通过 repeat_interleave 临时展开，高效 Attention Kernel 也可以直接按映射关系访问，不必物理复制完整历史。运行时可能有临时 Tensor，但它不是随上下文常驻的那份 KV Cache。

同理，GQA 不代表主 Attention 的 Q Head 计算量直接缩小为四分之一。它减少了 K/V 投影与历史数据的存储、读取，主 Attention 的 Q 输出头数仍然存在。

### 手写一个最小的 MHA/GQA/MQA

以下代码省略了 RoPE、QK-Norm、Dropout，仅用于验证 Head 映射和缓存形状，不用于生产推理：

~~~python
import math
import torch
import torch.nn as nn

class GroupedAttention(nn.Module):
    def __init__(self, hidden=512, q_heads=8, kv_heads=2):
        super().__init__()
        assert q_heads % kv_heads == 0
        self.q_heads, self.kv_heads = q_heads, kv_heads
        self.dh = hidden // q_heads
        self.group = q_heads // kv_heads
        self.q_proj = nn.Linear(hidden, q_heads * self.dh, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * self.dh, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * self.dh, bias=False)
        self.o_proj = nn.Linear(q_heads * self.dh, hidden, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.q_heads, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.dh).transpose(1, 2)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v  # 只有 kv_heads 组
        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)
        L = k.shape[2]
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        q_pos = torch.arange(L - T, L, device=x.device)[:, None]
        k_pos = torch.arange(L, device=x.device)[None, :]
        logits = logits.masked_fill(k_pos > q_pos, float("-inf"))
        out = torch.softmax(logits, dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))

for n_kv in (8, 2, 1):
    model = GroupedAttention(q_heads=8, kv_heads=n_kv).eval()
    cache = {}
    with torch.no_grad():
        _ = model(torch.randn(2, 16, 512), cache)
    print(n_kv, tuple(cache["k"].shape))

# 8 → (2, 8, 16, 64) MHA
# 2 → (2, 2, 16, 64) GQA
# 1 → (2, 1, 16, 64) MQA
~~~

两处细节值得记住：用 repeat 替换 repeat_interleave 可能破坏连续分组映射，而且不一定报错；某些模型独立指定 head_dim，不一定等于 hidden_size / num_attention_heads。原始 [GQA 论文](https://aclanthology.org/2023.emnlp-main.298/) 还讨论了如何从 MHA Checkpoint 经 KV 头均值池化与继续训练得到 GQA，这是模型训练/转换方法，而非推理时简单改参数即可无损转换。

## 3. MLA：和 GQA 省 KV 的思路有什么不同？

GQA 是减少完整的 K/V Head 数，**MLA（Multi-head Latent Attention）则换成一个低秩的共享 KV 表示**。

~~~text
MHA / GQA：每个历史 Token → 多组 K/V → KV Cache

MLA：每个历史 Token → KV 低秩压缩 → c_KV
                                     + k_R（独立 RoPE Key）
                                     → Latent KV Cache
~~~

DeepSeek 系列常见的 MLA 配置缓存 512 维的 KV latent 和 64 维 RoPE Key，即每 Token、每层 576 个元素。解码时可以通过权重吸收等手段在压缩表示上计算，不必先把每个 Head 的完整 K/V 全部展开到显存。解耦出部分 RoPE 信息，是为了让位置旋转不会阻碍相关线性变换的吸收。

**MLA 不是 Linear Attention。** 经典 Dense MLA 仍然对有效历史 Token 执行 Softmax Attention，只是把每个 Token 的 KV 保存得更紧凑；它并没有把完整历史改写成一个每步覆盖的固定大小状态。

它也解释了一个常见的分布式疑惑：**仅仅按 Attention Head 做 TP，往往不会自动沿序列分摊 MLA latent 历史。** 各个 Head 仍需访问同一份压缩 KV，多个 TP Rank 可能各自存储或读取完整历史。要把长历史进一步分到多个 GPU，就要引入 Context Parallelism / DCP。

## 4. Full、Sparse/DSA、Linear/KDA：历史究竟怎么被读取？

前面讨论的是缓存**存什么**，现在换一个问题：本步 Query 究竟**读哪些历史**？

<p align="center">
  <img src="/assets/posts/attention-full-dsa-kda.svg" width="100%" alt="Full Attention访问全部有效历史，DSA用Indexer选Top-k，KDA递推固定形状State">
</p>

*图 3 · 三种不同的读取与更新方式；不代表实测的性能倍数。*

**Full Attention**：新 Query 可访问所有有效历史位置，读取和注意力工作通常随 L 增长。

**SWA（Sliding Window Attention）**：只允许访问最近的一段固定窗口，以放弃部分远距离直接访问能力换取成本降低。

**DSA（DeepSeek Sparse Attention）**：先用较轻量的 Indexer 扫描历史，选择 Top-k，再对选中的 KV 做主 Attention。它保留了 Softmax，只是缩小主 Attention 的读取集合。不能说它“完全不再扫描全历史”，因为 Indexer 仍然需要打分。

**Linear Attention / KDA**：不再依赖每次 Decode 都重新逐 Token 读取整段 KV，而是随时间不断更新一个固定形状的 Recurrent State。下一步 Query 从这个 State 读取历史聚合结果。KDA 在这个递推框架里结合了门控遗忘与 Delta Rule，因此状态更新并不是简单 KV 平均。

~~~text
Full MLA：
    Qₜ → 读取 L 条 latent KV → Softmax → Output

DSA：
    Qₜ → Indexer 扫描 L 条 Index Key
       → Top-k → 读取 k 条 latent KV → Sparse MLA → Output

KDA：
    State(t-1) + 当前 Token → 更新 State(t)
    Query 从 State(t) 读取 → Output
~~~

DSA 主要减少昂贵的 MLA 数据读取，KDA 改变历史存储和更新范式；它们与 KV 压缩技术并不天然互斥。

## 5. Kimi K3 → K3.1：为什么同一个模型里既有 KDA 又有 DSA？

Kimi K3 总计 **93 层**，其中 **69 层 KDA + 24 层 Gated MLA**。绝大多数层用 KDA 的递推状态处理历史，一部分层仍通过 MLA 直接访问历史 Token。这类混合结构的思路是让固定状态与显式历史访问各自发挥作用。

到了 Kimi K3.1 Pro，模型仍保留 **69 层 KDA**，但另外 **24 层**由 Dense Gated MLA 变成 **Gated DSA（以 MLA 为基础的 Sparse Attention）**。根据 [Moonshot AI 的官方模型卡](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)，Indexer 为 32 Head × 128 维，Top-k = 2048。每个 DSA 层可以独立挑选当前 Query 更需要的历史位置。

<p align="center">
  <img src="/assets/posts/attention-kimi-k3-k31-memory.svg" width="100%" alt="K3的69个KDA层加24个Dense MLA层，K3.1保留KDA并将24层换成带Indexer的Sparse MLA/DSA">
</p>

*图 4 · K3/K3.1 的结构变化：不是 KDA 被 DSA 替代，而是 MLA 的全历史读取被稀疏化。*

~~~text
K3：
  69 个 KDA 层：更新每个请求的 Recurrent State
  24 个 MLA 层：读取全历史 latent KV

K3.1：
  69 个 KDA 层：继续更新 Recurrent State
  24 个 DSA 层：
      Indexer 扫全部 Index Keys
      → 选 Top-2048
      → 主 MLA 只读选中 latent KV
~~~

所以 **DSA 并不取代 MLA 的 KV 表示**，反而是在 MLA 旁边新增了一个 Indexer，以及由此产生的新缓存。这个变化从模型结构一路影响到显存规划。

### 为什么 K3.1 做 Sparse Attention，常驻 KV Cache 可能反而更大？

因为 **存储占用和每步读取带宽并不等价**。

在 [vLLM K3.1 Day-0 文章](https://vllm.ai/blog/2026-11-04-kimi-k31-pro) 的 FP8 口径下，每个 DSA 层每 Token 保存：

~~~text
MLA latent = 512 + 64 = 576 bytes
Index Key  = 128 bytes + 4 bytes scale = 132 bytes

K3     : 24 × 576 = 13,824 bytes / token
K3.1   : 24 × (576 + 132) = 16,992 bytes / token
~~~

也就是说，同为 FP8 缓存时，**K3.1 多出 3,168 bytes/token 的 Index Key 常驻开销**，却可能显著减少本次 Decode 实际读取的 latent KV：

~~~text
K3 Dense MLA 读取： 约 576 × L bytes / 层

K3.1 DSA 读取：    约 132 × L + 576 × min(2048,L) bytes / 层
                    ↑ Indexer       ↑ 主 MLA
~~~

例如 L=128K，按这套简化模型，Dense MLA 约读取 75.5 MB，DSA 的 Index Key 加选中 latent 约 18.5 MB，**数据读取量约为前者的四分之一**。但这不意味着 E2E TPOT 自动快四倍：Top-k 的计算、DCP 通信、KDA、MoE、CUDA Graph、Overlap 以及调度都会影响最终收益。缓存还可以对 latent 与 Index Key 分别采用 NVFP4、MXFP4 等压缩格式，容量比较必须标注具体数据类型。

## 6. KV Cache 和 Recurrent State：为什么服务端需要两种状态管理？

这是我觉得最能把 Attention 理论和 AI Infra 联系起来的地方。

在 Dense Attention 中，KV Cache 基本是**按 Token 追加**：新 Token 来了，追加一页新的历史；已经存在的 KV 通常不再原地修改。KDA 则维护**按请求分配、每步会被更新**的 Recurrent State。KDA 的状态还可能包含短卷积（Short Conv）所需的历史。

| 维度 | MLA/DSA KV 与 Index Key | KDA Recurrent State |
| --- | --- | --- |
| 分配粒度 | 按历史 Token 或 Page | 按活动请求 / 状态槽位 |
| 随上下文长度增长 | 通常线性增长 | 单份当前 State 形状固定 |
| 随并发增长 | 通常增长 | 活动请求越多，状态副本越多 |
| 更新 | Append | 原地更新 / Checkpoint |
| Prefix Cache | 可共享已经计算的 KV 页 | 需恢复对应前缀位置的 State |
| 推测解码 | 保证已验收的 KV 正确提交 | 还需保证失败分支的 State 回退 |
| P/D 分离 | 传递 paged KV / Index Key | 传递 KDA State |

[SGLang K3 Day-0 工程文章](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support) 给出了很直观的量级：在其 TP8 等指定条件下，69 层 KDA 的一份请求 State 约 **54 MB**，而跨 24 层的 MLA KV 是**按 Token 分配的小块**。这是该配置的参考值，不是所有精度和并行配置都相同。

于是混合模型可能出现一种很不符合传统直觉的情况：

~~~text
MLA KV Pool：还可以容纳很多 Token
KDA State Pool：已经没有新的活动请求槽位

结果：即使 KV Usage 不高，新的请求也不一定能 Admission
~~~

理解并发上限时，应当同时看：

~~~text
有效并发 ≤ min(
    KV Pool 支持的请求数,
    KDA State 支持的请求数,
    Runtime / Scheduler / 其他内存资源限制
)
~~~

这里还有一个很重要的区分：**“存在两种逻辑状态”不代表“所有框架都用两个独立物理内存池”。** SGLang 的 K3 Day-0 资料描述过 KDA State Pool 与 MLA KV Pool 分开预留造成的容量不匹配，也讨论统一分配方案；vLLM K3.1 的实现则把 MLA latent、Index Keys 和 KDA State 按 Block 打包进共享布局。具体策略必须以对应版本源码为准。

无论内存怎样合并，语义上的差异仍在：KDA State 是可变的，不能像只读历史 KV 那样直接被多个请求共享使用。Prefix Cache 命中时，除找到历史 KV 外，还需要对应位置的 **State Checkpoint**。SGLang 的方案包括 Copy-on-Write、Snapshot、Donate 等状态操作。Speculative Decoding 里又涉及失败分支的 State 恢复；ReplaySSM 则通过重放较小的中间数据，避免给每个候选步骤复制巨大的状态。

因此混合架构的容量问题并不只是“KV Cache 有多少 GB”，还要看状态槽位、前缀快照、状态复用和回滚机制。

## 7. DP-A 利用 MLA，DCP 再解决长上下文：为什么 MoE 还要走 EP？

如果只用一句话理解 DeepSeek / Kimi 这类 MLA + MoE 推理系统的混合并行：**Attention 用 DP-A 避免重复 KV，MoE 用 EP 分布专家；当单个 Attention 请求的上下文过长时，再用 DCP 把 KV 和读取工作拆到多个 Rank。**

这里的 **DP-A（Data-Parallel Attention）不是普通完整模型 DP**。在 SGLang 中，前者通过 attention DP groups 处理不同请求，后面的 MoE/FFN 仍可以按 TP/EP 协同计算；传统 DP 则是复制模型实例，让各实例独立完成整个 Forward。

### DP-A：把 MLA 的共享 latent 优势兑现成多请求并行

MLA 把不同 Head 需要的历史压缩为共享 latent KV。在普通 TP-only Attention 中，各 GPU 虽可切 Q Head，却可能需要保存和读取**同一请求的完整 latent 历史**。例如 TP8 下，同一份历史可能在 8 个 Attention TP Rank 上重复保存。

DP-A 不沿 Head 再切这份历史，而是**让不同 Rank/Attention 组处理不同请求**：Request A 的 MLA KV 放在其归属组，Request B 放在另一组。每组 Attention 能独立做 Prefill、Decode 或空闲，缓存无需在所有 Attention Rank 为同一个请求重复存储。这正是 MLA 特别适合 DP-A 的原因；但这不是 MLA 独有的数学要求，标准 Attention 模型也可以使用 DP-A。

MoE 层又是另一回事。Experts 通过 **EP（Expert Parallelism）** 分布在 GPU 上。Attention 后的 Hidden States 根据 Expert Routing 跨 Rank 发送，经过 **Dispatch → Expert GEMM → Combine** 回到各自的请求。MegaMoE 属于融合式 MoE 执行后端，能把 Dispatch 与 GEMM 融合来改善这条路径上的执行效率，**不是 DP-A 或 EP 的同义词**。

<p align="center">
  <img src="/assets/posts/attention-dpa-ep-megamoe.svg" width="100%" alt="同一组GPU上Attention采用DP-A，各Rank管理不同请求和MLA KV；MoE阶段经EP Dispatch、Expert Compute和Combine跨Rank执行，MegaMoE优化MoE执行路径">
</p>

*图 5 · 同一组 GPU 上，Attention 和 MoE 可以采取不同的并行组织方式。*

### DCP：解决“单个请求仍有很长的历史”这一瓶颈

这时候把 DCP 看成“给 DP-A 补短板”，直觉上很准确：

- **DP-A 按请求分**：解决同一份 MLA KV 被多个 TP Rank 重复存储，以及让不同请求独立调度的问题。
- **DCP 按历史 Token 分**：即使请求已经归属于某个 Attention 组，超长上下文的 latent KV 仍然随长度增长，单 Rank 的容量与长历史读取可能成为瓶颈。
- **DCP 把该请求的 Context 分片**到一个 DCP 组内的 GPU，各 Rank 对自己的历史分片算 Partial Attention，再通过 LSE（Log-Sum-Exp）完成正确的 Softmax 合并。

~~~text
TP-only MLA：
  Rank 0 : Request A 的全部 KV
  Rank 1 : Request A 的全部 KV
  Rank 2 : Request A 的全部 KV
  Rank 3 : Request A 的全部 KV
  → 同一份历史重复存储

DP-A：
  Attention Rank/组 0 : Request A 的 KV
  Attention Rank/组 1 : Request B 的 KV
  → 不同请求分开管理，KV 不再跨组重复

DP-A + DCP：
  Attention Group 0，处理 Request A：
       GPU 0 : A 的历史分片 0
       GPU 1 : A 的历史分片 1
             ↓ Partial Attention + LSE Merge
  Attention Group 1，处理 Request B：
       GPU 2 : B 的历史分片 0
       GPU 3 : B 的历史分片 1
             ↓ Partial Attention + LSE Merge
  → 先按请求分，再在所属组内按 Context 分
~~~

所以 **DCP 不是拿另外几个 DP-A Rank 的整份 KV 拼起来**，而是为同一个请求建立 Context Parallel 组。它降低每 Rank 的 KV 容量和长上下文读取成本，代价是增加跨 Rank 通信、部分结果合并，以及可能的 Query 复制/同步。是否值得开启，需要看长度、DCP 规模、硬件互联和关键路径。

一个不能省略的部署条件：**DCP 组必须放在兼容的 Attention TP 子组内**。例如 TP8 / DP-A4 / DCP2 是一种可理解的分组示意；但不能在只有 8 张卡、TP8 / DP-A8 已经每组 1 张卡的情况下，再凭空把 DCP8 套进去。具体还要符合目标 SGLang 版本的拓扑限制。

我们之前优化的 DCP A2A pack Kernel，正好位于 **Partial Attention 输出交换与合并** 所在的数据路径。它减少了局部打包 Kernel 的时间，但能否提升端到端 Decode，还要看 A2A、Attention 和其他算子是否处在关键路径。这与“DCP 可以省显存、却可能引入通信成本”并不矛盾。

### K3.1：DSA、DCP、KDA 如何同时存在？

K3.1 的 DSA 会先扫描 Index Keys，再做 Global Top-k，随后对选中的 MLA latent KV 执行 Sparse Attention。**DSA 决定访问哪些历史位置；DCP 决定这些位置由哪个 GPU 持有并计算；DP-A 决定由哪组 GPU 承接哪条请求。** 它们是不同维度：

~~~text
Request A → DP-A Attention Group
           → 各 DCP Rank 扫描自己持有的 Index Keys
           → 交换候选，形成 Global Top-2048
           → 对本地选中 KV 做 Partial Sparse MLA
           → Partial Output + LSE Merge
           → 得到 Request A 的 Attention Output
           → MoE Router → EP Experts → Combine
~~~

而 KDA 的 **Recurrent State** 具有按请求更新的生命周期，并不天然采用 MLA KV 的 DCP 历史分片方式。在 P/D 分离时也不能只传 KV：

~~~text
K3 Prefill → Decode：
  MLA Latent KV + KDA Recurrent State

K3.1 Prefill → Decode：
  MLA Latent KV + DSA Index Keys + KDA Recurrent State
~~~

因此，DCP 既是长上下文的容量/带宽优化，也是需要和 DP-A、EP、P/D 状态传输、Prefix Cache 一起设计的系统级并行策略。参考 [SGLang DP-A 指南](https://docs.sglang.ai/advanced_features/dp_dpa_router_guide.html) 和 [DCP 实现文档](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/dcp.mdx)。

## 8. 回到推理工程，看到模型配置应该先问什么？

如果现在再打开 SGLang 的模型配置与 Runtime，我会按照下面的顺序排查，而不是只看用了哪个 Attention 名字：

- **MHA/MQA/GQA**：先看 Hq、Hkv、head_dim 和 KV Cache 每 Token 的实际字节数；不要把 repeat_kv 的逻辑展开当作常驻缓存。
- **MLA**：检查 latent 维度、RoPE Key、Weight Absorption、TP 下是否复制，以及 DCP 是否沿历史分片。
- **DSA**：检查 Indexer 扫描长度、Top-k、Index Key 缓存、Global Top-k 通信和 Sparse Attention 的关键路径。
- **KDA/Linear**：检查 Recurrent State 的并发槽位、Prefix Cache Checkpoint、Spec Decode 回滚、P/D 传输和状态池占用。
- **K3/K3.1 混合服务**：同时核算 Token 级 KV 与 Request 级 State。前者可能限制最长上下文与 token capacity，后者可能先决定可接纳的并发。

最终，Attention 不是只决定模型如何计算注意力分数。它还决定推理引擎要存什么、如何调度内存、把哪些状态传到其他节点，以及端到端性能究竟卡在哪里。

**MHA/GQA 是 KV 头共享问题；MLA 是压缩表示问题；DSA 是稀疏读取问题；KDA 是递推状态问题。Kimi K3/K3.1 则把这些看似独立的概念放进了同一套可运行的系统。**

---

## 参考阅读

- [GQA · Ainslie et al. (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.298/)
- [DeepSeek-V3.2 Sparse Attention / Transformers](https://huggingface.co/docs/transformers/en/model_doc/deepseek_v32)
- [Kimi Linear 技术报告](https://yzhang.site/assets/pubs/techreport/2025/kda.pdf)
- [Moonshot AI · Kimi K3 官方技术博客](https://www.kimi.com/en/blog/kimi-k3)
- [Moonshot AI · Kimi K3.1 Pro 模型卡](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)
- [SGLang / LMSYS · K3 Day-0 与状态管理](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)
- [SGLang · Kimi K3 Cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx)
- [vLLM · K3.1 Pro Day-0（Indexer、Cache 布局与 DCP）](https://vllm.ai/blog/2026-11-04-kimi-k31-pro)
- [LightSeek · K3.1 Sparse Attention 工程实践](https://lightseek.org/blog/tokenspeed-kimi-k3-1)
