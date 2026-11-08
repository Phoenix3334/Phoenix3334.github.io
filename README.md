# Feihong Guo · AI Infrastructure and LLM Serving

[Live portfolio](https://phoenix3334.github.io/) · [SGLang contribution index](SGLANG_CONTRIBUTIONS.md) · [GitHub profile README](GITHUB_PROFILE.md)

## Baidu AI Cloud · AI Infrastructure internship, May–November 2026

- **Serving best practices.** Built reproducible SGLang benchmarks for DeepSeek-V4-Pro, V4.1-Flash and Kimi K3 on H200/B300/GB300. Compared input/output lengths, concurrency, parallel layouts, speculation, cache/state pool budgets, and CUDA Graph settings. Used MegaMoE microbenchmarks, nsys and ncu to connect component behavior with serving measurements.
- **Hardware selection, P/D sizing and deployment economics.** Evaluated throughput, TTFT/TPOT and memory limits at workload-specific operating points. Used per-stage QPS under a target SLO to estimate Prefill/Decode instance ratios and GPU requirements. The results feed comparative cost scenarios and TCO modeling.
- **Runtime debugging and upstream work.** Used source-level analysis, reproduction, controlled A/B and regression checks to isolate scheduling and memory-lifetime defects. DP health routing and PP tensor lifetime are the two main cases.
- **Engineering.** Maintained inference images and CI/CD workflows, validated multi-node environments, and automated dependencies, service lifecycle, benchmarking and regression collection.

## Selected SGLang contributions

| Record | My contribution | Result |
| --- | --- | --- |
| [#35721](https://github.com/sgl-project/sglang/pull/35721) | Isolated health-check routing state from user dispatch using a 2×2 controlled Prefill experiment | Authored · merged · v0.5.22 |
| [#47614](https://github.com/sgl-project/sglang/pull/47614) | Fixed cross-stream lifetime of received PP tensors with `record_stream` | Authored · merged · 51–98/192 → 0/192 anomalous requests in tested B300 fake-PD cases |
| [#43786](https://github.com/sgl-project/sglang/pull/43786) | Fused inverse RoPE into CUDA UE8M0 FP8 `wo_a` activation quantization while preserving BF16 rounding and packed scales | Authored · merged 4 Nov 2026 · B200 kernel and graph tests 7/7 |
| [#48404](https://github.com/sgl-project/sglang/pull/48404) | Reproduced incompatible EPLB recorder modes crashing at rebalance and added startup argument validation | Authored · merged 7 Nov 2026 · 4×B300 local invalid/valid path checks |
| [#39487](https://github.com/sgl-project/sglang/pull/39487) | DCP KV retraction report, draft and validation | Maintainer-integrated fix · merged |
| [#32467](https://github.com/sgl-project/sglang/pull/32467) | DSpark planner race investigation, causal validation and technical review | Fix by EanWang211123 · merged |
| [RFC #44697](https://github.com/sgl-project/sglang/issues/44697) | MoE imbalance profiling across real and synthetic routing and physical placement | Authored · open |

## Ascend and Kunpeng

- **Ascend 910B:** xLLM/vLLM-Ascend model deployment and runtime tuning, alongside Ascend C, Triton and TileLang kernel work.
- **MiniCPM-o 4.5 on Ascend 910C:** B300/DP8 DSpark draft training to GGUF deployment, ACL Graph, prefix KV reuse and TileLang integration. Paired 120 s duplex Core RTF **0.6754 → 0.4829**. [Project](https://github.com/Phoenix3334/minicpmo45-ascend).
- **Kunpeng 920:** SIMD, NUMA affinity, MPI/OpenMP and FFT planning. 256³ FFTW **80.97 → 555.00 GFLOP/s**.

## Measurement and cost boundaries

Phase-isolated, component and end-to-end measurements retain their workloads and test scopes. Historical backports and final upstream validation are reported separately. Full TCO requires actual prices, power, utilization, redundancy, operations and delivered traffic.

Method references: [Databricks inference performance engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) and [NVIDIA inference benchmarking and cost](https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/).

## Publication

GitHub Pages serves the repository root from `main`. The site is bilingual and requires no build step.
