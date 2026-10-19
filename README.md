# Feihong Guo — AI Systems / MLSys

Buildless, bilingual source for <https://phoenix3334.github.io/>.
English is the default; the language button switches to Chinese and remembers
the preference when browser storage is available.

## Structure

- Systems lens: physical workloads and compute/runtime foundations.
- Experience: Baidu AI Cloud, Huawei ICT Academy / ZZU Supercomputing,
  ZZU–Hanwei IoT Research Institute, and ThermoPulse.
- Internship notes: isolated P/D measurements, capacity estimation,
  and workload-dependent parallelism trade-offs.
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
- DSpark preserves the early authored failure reports #31023 / #33356 and\n  diagnostic/RFC/roadmap work. Phoenix3334 submitted a formal `COMMENTED`\n  GitHub Review on #32467; link both the Review record and the A/B/C inline\n  analysis. Present technical review as a contribution in this repair.\n- DSpark A/B/C validation uses 30,000 calls per variant. The review evidence
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
- EPLB is presented as a performance investigation without a validated throughput\n  gain. Placement-only evidence used no extra replicas; held-out evaluation is\n  incomplete and earlier request-path confounds are disclosed. SGLang #44697 and\n  DeepSeek EPLB #29 are authored open proposals/discussions, not shipped fixes or\n  evidence of a quality-preserving routing optimization.\n- Existing Issue/PR statuses were checked on 15 October 2026; the EPLB records\n  were checked on 16 October 2026. Public GitHub records are authoritative for\n  later changes.
- Physical and medical systems are prototypes; project evaluations do not
  establish field reliability or clinical effectiveness.
- Do not publish private scripts, hostnames, environment paths, model assets,
  internal code, credentials, or unpublished operational details.
