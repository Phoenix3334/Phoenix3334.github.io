# Feihong Guo — SGLang contributions

A portfolio index of **Feihong Guo / Phoenix3334's** authored fixes, technical reviews, and collaborative work in [sgl-project/sglang](https://github.com/sgl-project/sglang).

Development branches live in the [SGLang fork](https://github.com/Phoenix3334/sglang). This independent index stays in the portfolio repository so upstream synchronisation does not affect personal presentation.

[Portfolio](https://phoenix3334.github.io/) · [All authored SGLang PRs](https://github.com/sgl-project/sglang/pulls?q=is%3Apr+author%3APhoenix3334) · [Formal reviews](https://github.com/sgl-project/sglang/pulls?q=is%3Apr+reviewed-by%3APhoenix3334+-author%3APhoenix3334)

Status checked **8 November 2026**. This index lists **four authored merged SGLang PRs**, selected open work, and separately attributed technical reviews and collaborative fixes.

## Authored and merged

| PR | Problem and contribution | Validation boundary |
| --- | --- | --- |
| [#47614 — PP proxy tensor stream lifetime](https://github.com/sgl-project/sglang/pull/47614) | Traced output corruption at deeper PP queues to allocator reuse of received tensors before queued forward-stream reads. Added `record_stream(self.forward_stream)` at the shared PP launch boundary. **Merged 2 Nov 2026.** | B300 Kimi-K3 tests used v0.5.22 and fake PD transfer across PP2/PP4 and DCP combinations. Retained checksum probes support the lifetime diagnosis. Real two-node PD is outside the reported coverage; no throughput gain is claimed from corrupted baseline output. |
| [#35721 — DP health routing isolation](https://github.com/sgl-project/sglang/pull/35721) | Isolated health-check routing and load-budget bookkeeping from user dispatch. Authored implementation and regression tests. **Merged 11 Oct 2026; included in v0.5.22.** | Patched/unpatched × health-on/off B300 controls; four patched health-on repetitions recovered the expected throughput range. The rebased CPU file passed 30 tests; B300 evidence predates that rebase. |

| [#43786 — inverse RoPE + FP8 quantization](https://github.com/sgl-project/sglang/pull/43786) | Fused inverse RoPE into CUDA UE8M0 FP8 activation quantization, preserving BF16 rounding and packed scales. **Merged 4 Nov 2026.** | Upstream B200 numerical and replay kernel tests: 7/7. The historical B300 backport measured +1.61% Prefill throughput. Full-model performance of the final upstream implementation remains unmeasured. |
| [#48404 — EPLB recorder compatibility](https://github.com/sgl-project/sglang/pull/48404) | Reproduced an assertion at first rebalance and moved incompatible `per_pass` / `per_token` configuration rejection to startup. **Merged 7 Nov 2026.** | On the tested v0.5.23 local stack, 4×B300 verified startup rejection for the invalid mode and a successful rebalance/generate path with `stat`. |

The [DP Health case study](https://phoenix3334.github.io/#project-dp-health) walks through the Delayer hypothesis, measurement checks, request tracing and the same-image 2×2. The [public retrospective](https://github.com/sgl-project/sglang/issues/35241) preserves the chronology and separates the post-fix running-request analysis from the original discovery.

The [PP Stream case study](https://phoenix3334.github.io/#project-pp-lifetime) covers async-depth reproduction, receive-versus-forward checksums, the reference-lifetime control, and why `record_stream` protects allocator lifetime. [Original issue](https://github.com/sgl-project/sglang/issues/47613).

## Technical review and collaborative repairs

| Upstream record | My contribution | Authorship and status |
| --- | --- | --- |
| [#32467 — ragged-prefill plan race](https://github.com/sgl-project/sglang/pull/32467) | Early reports [#31023](https://github.com/sgl-project/sglang/issues/31023) / [#33356](https://github.com/sgl-project/sglang/issues/33356), reproductions, [formal COMMENTED review](https://github.com/sgl-project/sglang/pull/32467#pullrequestreview-4896206171), and [A/B/C inline analysis](https://github.com/sgl-project/sglang/pull/32467#discussion_r3749014153). Same-session 30,000-call variants: original 13,111 invalid plans; barrier and reviewer-proposed no-init variants both 0. | **Merged**, authored by **EanWang211123**. My work is investigation, causal validation, and technical review. |
| [#32183 — verifier state rewrite window](https://github.com/sgl-project/sglang/pull/32183) | Formal COMMENTED review clarifying that this is a separate correctness layer; I did not run an isolated A/B of this fix. | **Open**, authored by **slchenchn**. Review participation does not imply approval or validated speedup. |
| [#39487 — DCP retraction index localisation](https://github.com/sgl-project/sglang/pull/39487) | Reported [#38645](https://github.com/sgl-project/sglang/issues/38645), diagnosed the logical-to-local MLA KV mapping, implemented and validated [draft #38790](https://github.com/sgl-project/sglang/pull/38790). Unpatched DCP8 failed at first retraction; patched local stack completed 2,048 requests with 8 retractions. | **Merged**, integrated by **kpham-sgl** with shared layout and compatibility handling. #38790 closed as superseded; local results are not tests of the final upstream head. |

## Authored open work

| PR | Focus | Current status |
| --- | --- | --- |
| [#45384](https://github.com/sgl-project/sglang/pull/45384) | Lazy CUDA VMM multimodal transport initialisation | Open |
| [#45584](https://github.com/sgl-project/sglang/pull/45584) | Serialise multi-tokenizer general warmups | Open; distinct from lazy transport initialisation |
| [#44472](https://github.com/sgl-project/sglang/pull/44472) | Detect MNNVL fabric independently of GPU names | Draft |
| [#34286](https://github.com/sgl-project/sglang/pull/34286) | Breakable CUDA Graph warmup-to-capture ordering | Open |
| [#34410](https://github.com/sgl-project/sglang/pull/34410) | DSpark P/D decode draft-input handoff regression | Open; regression coverage, not a new support claim |

## Serving diagnostics and investigation history

- [RFC #44697](https://github.com/sgl-project/sglang/issues/44697): production expert placement and synthetic routing diagnostics for Prefill. The real-token V4.1-Flash follow-up records rank-count CV, MegaMoE duration, input throughput and GPU timelines separately.
- [Issue #45367](https://github.com/sgl-project/sglang/issues/45367): two-node GB300 DCP graph replay investigation. The [registration guard #45371](https://github.com/sgl-project/sglang/pull/45371) and [persistent buffer proposal #45824](https://github.com/sgl-project/sglang/pull/45824) were closed unmerged.
- [#48514](https://github.com/sgl-project/sglang/pull/48514): test import adjustment, closed unmerged.

- [#31455](https://github.com/sgl-project/sglang/pull/31455): closed, unmerged diagnostic draft for DSpark replay synchronisation.
- [#38790](https://github.com/sgl-project/sglang/pull/38790): closed, unmerged implementation draft superseded by #39487.
- [RFC #32432](https://github.com/sgl-project/sglang/issues/32432) and [Roadmap #34297](https://github.com/sgl-project/sglang/issues/34297): runtime contracts, failure separation, and investigation map.

Public PR discussions provide the stack, workload, and evidence limits. No closed unmerged draft or another author's merged PR is counted as my authored merged PR.
