---
layout: post
title: "SGLang DP Health：一次没有报错的吞吐退化，根因却在请求路由"
date: 2026-11-18 11:10:00 +0800
lang: zh-CN
summary: "从 B300 上偶发的 Prefill 吞吐下降出发，复盘 generation-health 如何污染 DPC 用户路由状态，以及如何通过 2×2 受控实验确认并修复。"
tags: [SGLang, DP, Performance, Debugging]
---

同一个模型、同样的压测配置，为什么某些运行能稳定接近 **95k input tok/s**，另一些运行却只剩下 **68k**？更麻烦的是：请求全部完成，服务没有崩溃，也没有某个明显变慢的 GPU Kernel。

我在排查 SGLang 的一次 Prefill 吞吐异常时，最后定位到的根因并不在 Attention、MoE 或 CUDA Kernel，而是在更靠前的 **DataParallelController（DPC）**：健康检查请求错误地推进了属于真实用户流量的路由状态。

修复对应 [SGLang PR #35721](https://github.com/sgl-project/sglang/pull/35721)，原始问题见 [Issue #35241](https://github.com/sgl-project/sglang/issues/35241)。这篇文章记录的不只是一个修复，更是一次**从吞吐症状追到控制面状态，再用受控实验确认因果**的过程。

## 01 · 一个看起来不像 Bug 的性能问题

测试环境是 DeepSeek-V4-Pro、8×B300，使用 TP8/DP8、DP Attention、MegaMoE 与 PrefillDelayer。工作负载固定为 **60k 输入 / 1 输出、C16、160 个请求**，主要用于观察 Prefill 处理能力。

异常运行里，系统没有报错，请求也全部返回；真正不正常的是**各 DP rank 拿到的用户请求，以及最终形成的有效 batch**。正常情况下，用户任务在 rank 之间较均匀；LOW 运行里，per-rank work inventory 出现偏斜，有效 batch 缩小，整体吞吐随之下降。

这个观察改变了排查顺序。如果是纯算子退化，我预期相同工作量下每次执行都更慢；但这里更像是**工作进入 GPU 之前，分配方式就已经发生变化**。

## 02 · 一条在 Scheduler 之前出错的请求路径

我们可以把相关请求生命周期压缩成：

~~~text
Client / Generation-Health
          |
          v
    TokenizerManager
          |
          v
 DataParallelController    ← 选择 DP rank / 管理路由状态
          |
          v
  Scheduler / Delayer
          |
          v
  TP Worker / Model Runner
          |
          v
        GPU
~~~

DPC 的工作不是执行模型，而是决定请求应该交给哪个 DP rank。不同负载均衡策略需要不同状态：round-robin 依赖当前分发位置，负载感知策略可能还需要用户请求相关的 load budget。

**generation-health 请求也会经过 DPC，但它在后续 Scheduler 阶段并不成为等价的业务工作。** 旧实现里，这类健康检查可能推进共享的 user-RR 状态，或者触发 user-LB budget 更新。于是：

> 健康检查没有增加等量的用户计算，却改变了之后真实请求的分发历史。

结果不是健康检查直接吃掉了全部吞吐，而是共享路由状态被“伪用户工作”扰乱，真正的用户流量在 rank 间分发不均。最终，异常从控制面一路放大到了 GPU 的有效计算批量。

<p align="center">
  <img src="/assets/posts/dp-health-routing.svg" width="100%" alt="DP health check 修改共享用户路由状态，以及隔离后的路由结构">
</p>

这里有一个值得强调的区分：**Health 请求占了多少计算资源**，和 **它是否有权改变业务流量的路由状态**，是完全不同的问题。这次的主要根因是第二个。

## 03 · 怎样证明是它，而不是某个偶然相关的开关

仅凭“关掉 health 后恢复了”还不够：关闭一项功能也可能改变系统负载、CPU 开销或某个调度时序。更可靠的办法是让**补丁状态**与 **generation-health 开关**交叉组合，进行 2×2 对照。

这四组使用相同的用户负载与服务配置，每个实验组都重新启动服务。外部 HTTP health polling 仍然存在，所谓 Health OFF 主要隔离的是 generation-health 行为，而不是把全部探活流量删除。

| 代码与开关 | 实验结果 | Input tok/s |
| :--- | :--- | ---: |
| 原版 + Health ON | **2/8 次 LOW**，其余正常 | LOW 例子：68,403 / 82,883 |
| 原版 + Health OFF | 4/4 GOOD | 94,783–94,847 |
| 修复版 + Health ON | **4/4 GOOD** | 94,558–94,699 |
| 修复版 + Health OFF | 4/4 GOOD | 94,850–94,934 |

最有价值的不是某一次的最高 tok/s，而是 **LOW 只出现在原版 + Health ON 这一象限**；而修复版在保留 generation-health 的情况下，依然能保持正常路由与吞吐。

结合 DPC 的 health 计数、per-rank user dispatch 和 batch 观测，可以把因果链表达为：

~~~text
generation-health 进入 DPC
             ↓
共享 user RR / LB 状态被修改
             ↓
后续真实用户请求的 rank ownership 偏斜
             ↓
各 rank 有效 batch 下降或分化
             ↓
Prefill input TPS 间歇性回退
~~~

这比简单地说“Health ON 会降速”准确得多。实际上原版 Health ON 的八次运行中仍有六次正常；这是一个**条件触发、间歇放大**的 Bug，而不是稳定存在的固定额外开销。

## 04 · DP Attention 和 PrefillDelayer 是什么角色

这次现象发生在 DP Attention、长输入 Prefill 与 PrefillDelayer 的组合中。它们确实影响了异常的表现，但不能把这些组件直接当成根因。

DP Attention 让各 rank 的 Attention 工作具有相对独立的推进节奏；MoE 等阶段又涉及跨 rank 的协作。PrefillDelayer 参与请求推进与批次组织。当用户流量先在 DPC 处发生偏斜，后续调度更难维持均匀的有效工作量，吞吐异常就可能被放大。

我后来更重视**必要条件与伴随现象**的区分。PR 的后续分析表明，某些容易联想到的指标并不具有判别力：例如 <code>IDLE-while-EXTEND</code> 不是 LOW 必备事件，出现 <code>wait_timeout</code> 也不必然意味着 LOW。因此，不能把 Delayer 的 timeout 简化成这次 Bug 的根因。

更准确的三层关系是：**触发发生在 DPC 共享状态 → 分发失衡体现在用户工作量 → 下游执行和批次组织决定退化被放大的程度**。

## 05 · 修复不是关闭探活，而是隔离状态所有权

补丁的核心是一条很清楚的约束：

> generation-health 可以被路由，但不能更新真实用户请求的 RR / LB 状态。

实现上，健康检查使用独立的 RR 计数器；health-only dispatch 不再刷新用户 load budget。混合 batch 只有确实包含真实用户工作时，才更新对应用户预算。显式路由 rank、active worker 筛选和 P/D 相关语义也必须保持兼容。

这比在外层简单“少发健康检查”更可靠，因为它明确了 **State Ownership：状态归谁所有，谁可以修改它**。

验证同样分层看待：历史 B300 受控实验说明了具体故障族的性能因果；最终 rebased 代码的 **30 项 CPU 测试**证明相关路由与边界语义得到回归覆盖。需要诚实区分：**最终 rebase 之后没有重做同一轮 B300 性能测试**，因此不能把不同阶段的证据说成同一次实验。

## 06 · 从这次 Bug 学到的排障顺序

做推理系统优化时，很容易直接进入 nsys、ncu、NCCL、Attention 或 MoE 分析。但高性能 Serving 的“快”不仅要求 GPU 算得快，还要求**真实用户请求被正确分配，能持续形成有效 batch**。

这次之后，我遇到端到端吞吐下降，会先分辨：是工作量变化、工作分配变化、批次组织变化，还是执行同一批工作真的变慢？这四类问题的第一证据往往不一样。

**这次修复的核心不是让 health check 更轻，而是阻止它越过用户路由状态的边界。** 一个控制面上的小状态错误，最终可以呈现为 GPU 端的大吞吐退化。

---

**参考与代码**：[PR #35721](https://github.com/sgl-project/sglang/pull/35721) · [Issue #35241](https://github.com/sgl-project/sglang/issues/35241)

*实验边界：以上数字来自对应 DeepSeek-V4-Pro、8×B300、60k/1、C16/N160 的历史部署配置；不能推广为所有模型和所有 DP 拓扑的通用吞吐结论。*
