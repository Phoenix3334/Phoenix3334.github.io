<div align="center">

<img src="https://raw.githubusercontent.com/Phoenix3334/Phoenix3334.github.io/main/assets/profile/terminal-banner.svg" width="100%" alt="Feihong Guo / Phoenix3334 — AI Infrastructure and LLM Inference Engineering">

[Portfolio](https://phoenix3334.github.io/) · [Upstream contributions](https://github.com/Phoenix3334/Phoenix3334.github.io/blob/main/SGLANG_CONTRIBUTIONS.md) · [Authored SGLang PRs](https://github.com/sgl-project/sglang/pulls?q=is%3Apr+author%3APhoenix3334) · [Technical reviews](https://github.com/sgl-project/sglang/pulls?q=is%3Apr+reviewed-by%3APhoenix3334+-author%3APhoenix3334)

</div>

### `$ whoami`

**Feihong Guo · 郭飞鸿** — AI Infrastructure / LLM Inference Engineering  
Zhengzhou University · AI Infra intern at Baidu AI Cloud · SGLang contributor

I work on **SGLang serving performance and runtime correctness** for DeepSeek-V4-Pro and Kimi K3 on B300/GB300. My focus is connecting reproducible measurements and causal debugging to capacity, resource, and deployment-cost decisions.

### `$ work --focus`

| Track | What I work on |
| --- | --- |
| **Baidu / SGLang** | P/D benchmarking, DP/TP/EP/PP/DCP layouts, KV and graph capacity, speculative execution, runtime failures, and upstream regressions |
| **Huawei ICT / Ascend** | xLLM and vLLM-Ascend, Ascend C / Triton / TileLang, kernel-to-engine integration, and MiniCPM-o on Ascend 910C |
| **Capacity & economics** | Latency–throughput trade-offs, workload-specific capacity models, GPU allocation, and the inputs needed for TCO estimation |
| **CPU foundations** | Kunpeng HPCG / FFTW, memory locality, NUMA placement, SIMD, and MPI/OpenMP |

### `$ upstream --evidence`

| Record | My contribution | Upstream result |
| --- | --- | --- |
| [SGLang #47614](https://github.com/sgl-project/sglang/pull/47614) | PP receive-tensor lifetime diagnosis, implementation, and B300 validation | **Authored · merged** |
| [SGLang #35721](https://github.com/sgl-project/sglang/pull/35721) | Isolate health-check state from user routing; regression tests | **Authored · merged · v0.5.22** |
| [SGLang #32467](https://github.com/sgl-project/sglang/pull/32467#discussion_r3749014153) | Formal technical review and controlled A/B/C validation of the planner-race repair | **Fix merged** · EanWang211123 authored the PR |
| [SGLang #39487](https://github.com/sgl-project/sglang/pull/39487) | Original report, draft implementation, and DCP KV-recovery validation | **Fix merged** · kpham-sgl integrated the repair |
| [SGLang #32183](https://github.com/sgl-project/sglang/pull/32183#pullrequestreview-4897392620) | Formal review distinguishing verifier-window correctness from other DSpark failures | **PR open** |

<sub>PR states verified on 2 November 2026. Authored fixes, technical reviews, and maintainer-integrated repairs retain their own attribution.</sub>

### `$ investigate --current`

- **Attention kernels:** [inverse-RoPE + FP8 quantisation fusion #43786](https://github.com/sgl-project/sglang/pull/43786).
- **Graph-captured communication:** [DCP registration guard #45371](https://github.com/sgl-project/sglang/pull/45371) and [persistent buffers #45824](https://github.com/sgl-project/sglang/pull/45824), alternative proposals for the reported failure.
- **Multimodal startup:** [lazy CUDA VMM transport #45384](https://github.com/sgl-project/sglang/pull/45384) and [multi-tokenizer warmup serialisation #45584](https://github.com/sgl-project/sglang/pull/45584).

These proposals remain open; the linked discussions record the tested stacks and remaining validation.

### `$ projects --selected`

<a href="https://phoenix3334.github.io/#opensource"><img src="https://raw.githubusercontent.com/Phoenix3334/Phoenix3334.github.io/main/assets/profile/sglang-work.svg" width="49%" alt="SGLang upstream work — authored fixes, reviews, and collaborative repair"></a>
<a href="https://github.com/Phoenix3334/minicpmo45-ascend"><img src="https://raw.githubusercontent.com/Phoenix3334/Phoenix3334.github.io/main/assets/profile/minicpm-work.svg" width="49%" alt="MiniCPM-o / Ascend 910C — training-to-runtime engineering"></a>

- **[SGLang engineering portfolio](https://phoenix3334.github.io/#opensource):** serving measurements, root-cause investigations, merged repairs, and public evidence.
- **[MiniCPM-o / Ascend 910C](https://github.com/Phoenix3334/minicpmo45-ascend):** draft training, speculative execution, kernel fusion, and duplex profiling. Matched-harness Core RTF **0.6754 → 0.4829**; workload trade-offs and accuracy gates are documented.
- **[AI Infra portfolio source](https://github.com/Phoenix3334/Phoenix3334.github.io):** performance → capacity → resource and cost decisions, with Databricks / NVIDIA methodology references.

### `$ toolchain`

| Layer | Tools and systems |
| --- | --- |
| Serving & runtime | SGLang · xLLM · vLLM-Ascend · llama.cpp-omni · PyTorch |
| Kernels & execution | CUDA · Triton · TileLang · Ascend C · CANN · CUDA / ACL Graph |
| Profiling & validation | Nsight Systems · Nsight Compute · msprof · controlled A/B experiments · regression tests |
| Systems | Python · C/C++ · Linux · NCCL · MPI/OpenMP · NUMA |

### `$ method`

**Deploy → measure → explain → improve → validate → size.**

I evaluate throughput together with latency, correctness, failure behaviour, and memory limits. Phase-isolated, kernel-level, and end-to-end results keep separate measurement boundaries. Performance and capacity evidence support cost assessment; a complete TCO result also needs actual prices, utilisation, redundancy, operations, and delivered traffic.

### `$ contact`

[GitHub](https://github.com/Phoenix3334) · [Website](https://phoenix3334.github.io/) · [Email](mailto:1187883824@qq.com)
