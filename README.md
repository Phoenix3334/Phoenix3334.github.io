# Feihong Guo — AI Systems / MLSys

Buildless, bilingual source for <https://phoenix3334.github.io/>.
English is the default; the language button switches to Chinese and remembers
the preference when browser storage is available.

## Structure

- Systems lens: physical workloads and compute/runtime foundations.
- Experience: Baidu AI Cloud, Huawei ICT Academy / ZZU Supercomputing,
  ZZU–Hanwei IoT Research Institute, and ThermoPulse.
- Internship notes: sanitized B300/GB300 P/D measurement summaries, capacity
  estimation, component baselines, and workload-dependent parallelism trade-offs.
- Open-source evidence: model–system co-design, proposed fixes,
  diagnostic work, and collaborative validation.
- Research: adaptive inference as an open AI Systems / MLSys direction.

## Project visuals

Five project photographs/output images and the xLLM / vLLM-Ascend marks
were extracted from the supplied interview presentation. The WebP assets
preserve the source pixel dimensions, alpha channel, and decoded pixels.
Cutout figures sit directly on the page background without white containers
or decorative frames. Figure groups and captions are centered within the
content width. The inspection robot uses the original PPT right-crop viewport
(27.632%); the linked full image stays unchanged. Project figures link to
their full image and load lazily. They illustrate
project prototypes; the nearby text defines contribution and validation scope.

## Publish

GitHub Pages deploys this existing repository from `main`, repository root.
Keep `index.html`, `og.jpg`, and `.nojekyll` at the root. No build step or
package installation is required.

## Evidence and public-content policy

- Retain workload, hardware, baseline, and harness conditions with metrics.
- Prefill/Decode measurements are phase-isolated; Decode uses simulated
  speculative acceptance. They are not an end-to-end P/D result.
- Qwen combined-configuration gains and Gemma migration comparisons are
  independent measurements.
- MiniCPM training acceptance, profiled launches, and duplex RTF are separate
  endpoints. The final workload and latency trade-offs are stated explicitly.
- Feature authored fixes and public investigations alongside collaborative merged
  fixes. Distinguish authored production code, diagnostic tooling, independent
  validation, review evidence, and maintainer-authored integration. Closed,
  unmerged drafts may be linked as historical supporting records.
- DSpark preserves the early authored failure reports #31023 / #33356 and
  diagnostic/RFC/roadmap work. Phoenix3334 submitted a formal `COMMENTED`
  GitHub Review on #32467; link both the Review record and the A/B/C inline
  analysis. Present technical review as a contribution in this repair.
- DSpark A/B/C validation uses 30,000 calls per variant. The review evidence
  supports the final no-init implementation in merged #32467, authored by
  EanWang211123. The 18,560-request regression used the disclosed v0.5.16
  backport stack; forced-ragged coverage is a separate endpoint.
- DP #35721 was independently authored and merged, and is listed under
  Parallelism & Disaggregation in the [SGLang v0.5.22 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22).
  Initial work added 23 CPU tests; the rebased test file passed 30 tests.
  B300 runs predate that rebase.
- DCP #38645 and authored draft #38790 record discovery, root-cause analysis,
  implementation, and B300 validation. Maintainer kpham-sgl integrated the fix
  with shared layout/compatibility handling in merged #39487; the draft was
  explicitly closed as superseded. Local results are not final-PR-head tests.
- EPLB is presented as a performance investigation without a validated throughput
  gain. Placement-only evidence used no extra replicas; held-out evaluation is
  incomplete and earlier request-path confounds are disclosed. SGLang #44697 and
  DeepSeek EPLB #29 are authored open proposals/discussions, not shipped fixes or
  evidence of a quality-preserving routing optimization.
- Featured PR states were rechecked on 23 October 2026. #35721, #39487 and
  #32467 are merged; #34286, #34410, #43786, #45371 and #45824 remain open.
  Public GitHub records are authoritative for later changes.
- Physical and medical systems are prototypes; project evaluations do not
  establish field reliability or clinical effectiveness.
- Do not publish private scripts, hostnames, environment paths, model assets,
  internal code, credentials, or unpublished operational details.


## Measurement update — 23 October 2026

- DeepSeek-V4-Pro, B300: 60k input / 1k output AR baseline 32,879.35 to
  selected EAGLE configuration 46,806.57 output tok/s (+42.4%). The 128k
  comparison is 20,344.06 to 36,487.70 output tok/s (+79.4%). These combine
  SGLang 0.5.14 to 0.5.22, speculation and capacity/concurrency changes.
- Speculative acceptance is fixed and simulated at about 0.70, with mean
  accepted length about 3.10. These are not natural draft-quality measurements.
- GB300, two nodes / eight GPUs: 48,006.17 output tok/s at 60k/1k on the
  selected tuned SGLang 0.5.17 configuration. It is a separate operating point,
  not a matched hardware speedup against the B300 baseline.
- Prefill on SGLang 0.5.22: B300 95,440 input tok/s at 60k; GB300 117,830
  at 60k and 100,268 at 128k, all with one output token. Separate GB300 CP8
  low-latency points have median TTFT 0.666 s / 1.516 s at 60k / 128k.
- Results inform resource efficiency and deployment-cost assessment; no
  measured full-cost or TCO reduction is claimed. Larger capacity and newer
  versions are not presented as universally better.
- MegaMoE 1.51–1.87× and small-batch HBM about 5.1 TB/s describe separate
  component test points. Authored attention fusion #43786 has older +1.63%
  whole-model evidence; its cleanup refactor still requires GPU revalidation.
- GB300 DCP #45371 / #45824 are alternative authored fixes. #45824 head
  9cb1de5 has a 12/12-correct two-node probe and 23 tests + 5 subtests locally.
  The separate guard-policy 640/640 run is not a test of its later merged-up head.
- Kunpeng FFTW is corrected to 80.965 → 555.001 GFLOP/s (6.85×, 256³).

The public page retains hardware families, model names, workload lengths,
strategy categories and representative aggregates. It omits internal document
links, host/container identifiers, addresses, private registries, absolute paths,
raw logs, full launch recipes and unpublished best-capacity parameter values.
