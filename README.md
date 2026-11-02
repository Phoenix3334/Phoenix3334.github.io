# Feihong Guo — AI Infrastructure / LLM Inference

[Live portfolio](https://phoenix3334.github.io/) · [SGLang contribution index](SGLANG_CONTRIBUTIONS.md)

The portfolio follows one engineering thread: **deploy → measure → explain → improve → validate → size**.

## Focus

1. **Baidu AI Cloud / SGLang.** DeepSeek-V4-Pro and Kimi K3 on B300/GB300; Prefill/Decode evaluation, runtime correctness, cache/graph capacity, profiling, and upstream contributions.
2. **Serving performance.** Workload-specific measurements, latency/throughput trade-offs, concurrency and configuration matrices, capacity estimates, and bottleneck evidence.
3. **Huawei ICT Academy / ZZU Supercomputing.** Ascend 910B inference with xLLM/vLLM-Ascend, Ascend C/Triton/TileLang kernels, Kunpeng HPC studies, and a complementary MiniCPM-o / Ascend 910C runtime project.
4. **Capacity and TCO.** Use a qualifying operating point to estimate instance and GPU requirements, then account for deployment costs and delivered token volume.
5. **Engineering practice.** Reproducible workload definitions, profiler-guided attribution, causal experiments, correctness gates, and resource decisions.

## Selected upstream records

| Record | My contribution | Result |
| --- | --- | --- |
| [SGLang #47614](https://github.com/sgl-project/sglang/pull/47614) | PP proxy-tensor lifetime diagnosis, implementation, and B300 validation | Authored; merged 2 Nov 2026 |
| [SGLang #35721](https://github.com/sgl-project/sglang/pull/35721) | DP health-routing isolation and regression tests | Authored; merged; v0.5.22 |
| [SGLang #32467](https://github.com/sgl-project/sglang/pull/32467#discussion_r3749014153) | Formal review and controlled A/B/C planner validation | Fix merged; EanWang211123 authored the PR |
| [SGLang #32183](https://github.com/sgl-project/sglang/pull/32183#pullrequestreview-4897392620) | Formal review separating correctness layers | PR open |
| [SGLang #39487](https://github.com/sgl-project/sglang/pull/39487) | Original report, draft repair, and validation | Fix merged; kpham-sgl integrated the repair |

Statuses checked 2 November 2026. Authored fixes, technical reviews, and maintainer-integrated work retain separate attribution.

## Measurement and cost boundaries

- P/D-isolated results are labelled by phase, workload, hardware, and software stack.
- The displayed AR-to-EAGLE comparison combines version, speculation, and capacity changes; speculative acceptance is fixed and simulated.
- Kernel, standalone runtime, and end-to-end results are separate measurements.
- Local/backported validation is distinguished from tests of final upstream heads.
- Capacity estimates and measurements support resource selection and cost scenarios. Complete TCO additionally needs actual prices, utilisation, redundancy, operations, and delivered traffic.
- No measured production cost reduction is claimed from phase-isolated throughput alone.

## Methodology references

- [Databricks — LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [NVIDIA — LLM Inference Benchmarking: How Much Does Your LLM Inference Cost?](https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/)

These inform the methodology; benchmark results remain tied to the documented project workloads.

## Source and publication

Buildless, bilingual HTML/CSS/JavaScript. English is the default; the language switch remembers the visitor's choice when browser storage is available.

GitHub Pages deploys `main` from the repository root. Keep `index.html`, `og.jpg`, and `.nojekyll` in place. No package installation or build step is required.

Personal presentation lives in this independent repository. The SGLang fork remains an upstream development workspace.
