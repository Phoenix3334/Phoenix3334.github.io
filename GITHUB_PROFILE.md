# Feihong Guo · 郭飞鸿

**AI Systems / LLM Inference Infrastructure**

Undergraduate at Zhengzhou University. My work focuses on distributed LLM serving, runtime correctness, and GPU/NPU performance engineering.

[Portfolio](https://phoenix3334.github.io/) · [SGLang contributions](https://github.com/Phoenix3334/sglang/blob/main/CONTRIBUTIONS.md) · [Authored PRs](https://github.com/sgl-project/sglang/pulls?q=is%3Apr+author%3APhoenix3334) · [Technical reviews](https://github.com/sgl-project/sglang/pulls?q=is%3Apr+reviewed-by%3APhoenix3334+-author%3APhoenix3334)

## Selected upstream work

| Contribution | My role | Upstream result |
| --- | --- | --- |
| [SGLang #47614](https://github.com/sgl-project/sglang/pull/47614) — PP proxy-tensor lifetime across CUDA streams | Root-cause analysis, implementation, B300 validation | **Merged · 2 Nov 2026** |
| [SGLang #35721](https://github.com/sgl-project/sglang/pull/35721) — isolate health checks from user routing | Independently authored fix and regression tests | **Merged · v0.5.22** |
| [SGLang #32467](https://github.com/sgl-project/sglang/pull/32467#discussion_r3749014153) — ragged-prefill planner race | Formal technical review and controlled A/B/C kernel validation | **Fix merged** · authored by EanWang211123 |
| [SGLang #39487](https://github.com/sgl-project/sglang/pull/39487) — DCP retraction KV-index mapping | Discovery, diagnosis, [draft implementation and validation](https://github.com/sgl-project/sglang/pull/38790) | **Fix merged** · integrated by kpham-sgl |

I also submitted a [formal review on #32183](https://github.com/sgl-project/sglang/pull/32183), separating verifier-window correctness from the independently tested DSpark failure mechanisms.

## Current engineering work

- **Distributed serving:** DP attention, pipeline parallelism, P/D disaggregation, KV-cache recovery, and CUDA Graph / communication buffer lifetime.
- **Kernel and runtime optimisation:** [inverse-RoPE + FP8 quantisation fusion](https://github.com/sgl-project/sglang/pull/43786), [DCP a2a buffer lifetime](https://github.com/sgl-project/sglang/pull/45824), and [multimodal startup memory](https://github.com/sgl-project/sglang/pull/45584). These PRs remain open.
- **Cross-platform systems:** [MiniCPM-o 4.5 on B300 and Ascend 910C](https://github.com/Phoenix3334/minicpmo45-ascend) — draft training, speculative decoding, TileLang kernel fusion, and real-time duplex profiling.
- **HPC foundations:** Kunpeng CPU optimisation, NUMA placement, mixed MPI/OpenMP parallelism, HPCG, and FFTW.

## Featured repositories

| Repository | What to look at |
| --- | --- |
| [sglang](https://github.com/Phoenix3334/sglang) | My contribution fork and upstream evidence index |
| [minicpmo45-ascend](https://github.com/Phoenix3334/minicpmo45-ascend) | Training-to-runtime engineering; matched-harness duplex RTF 0.6754 → 0.4829, with workload trade-offs documented |
| [llama.cpp-omni-OpenBMB](https://github.com/Phoenix3334/llama.cpp-omni-OpenBMB) | Upstream development branches for [vision projection #144](https://github.com/tc-mb/llama.cpp-omni/pull/144) and [Flow / CPU Vocoder overlap #146](https://github.com/tc-mb/llama.cpp-omni/pull/146); #146 remains Draft |
| [Phoenix3334.github.io](https://github.com/Phoenix3334/Phoenix3334.github.io) | Bilingual project portfolio, measurements, and contribution records |

I work from **reproduction → causal experiment → correctness → performance → upstream review**. Benchmarks retain their workload and configuration boundaries; review and collaborative repairs are attributed separately from authored merged PRs.

<sub>Contribution states checked on 2 November 2026. GitHub PR pages are authoritative for later changes.</sub>
