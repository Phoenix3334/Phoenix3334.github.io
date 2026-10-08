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
- Feature authored issue reports, investigation roadmaps, RFCs, and open PRs.
  Closed PRs are not listed as featured contributions. Distinguish problem
  discovery, controlled evidence, proposed contracts, and authored patches.
- DSpark A/B/C validation uses 30,000 calls per variant; candidate validation
  and sustained runtime regression are separate evidence endpoints.
- The DCP issue is a closed public investigation record with local validation;
  its closed status is not presented as proof of a merged upstream fix.
- Issue and PR status is a dated snapshot, checked on 8 October 2026; public GitHub
  records are authoritative for later changes.
- Physical and medical systems are prototypes; project evaluations do not
  establish field reliability or clinical effectiveness.
- Do not publish private scripts, hostnames, environment paths, model assets,
  internal code, credentials, or unpublished operational details.
