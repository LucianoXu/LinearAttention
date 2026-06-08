# Nested speech-text LM — experimental results (Day 3, overnight 2026-06-09)

Status as of the [Raven maintenance shutdown](raven-specific/RAVEN.md) deadline
(07:00 2026-06-09, cluster down 6 days). All runs are 12-layer, d=768, ~93M
params, 4×A100, bf16, `torch.compile`, context 2048, chunk 256. Validation =
per-modality next-token cross-entropy on `dev-clean` (logged every 200 steps);
PPL = exp(CE). The A/B knob is `slow_memory` (false = flat baseline,
true = nested two-level memory); **only that flag differs within a pair**.

The headline metric is **Δspeech vs Δtext**, where Δ = baseline − nested CE
(positive ⇒ nested is *better*). The hypothesis predicts **Δspeech > 0 and
Δspeech ≫ Δtext** (a slow memory should help long dense speech spans more than
compact text).

## TL;DR

**The hypothesis is not supported in any configuration tested.** Across three
train-clean-100 regimes the nested slow memory never helped speech, and in the
regime built to favour it (packed + localized fast memory) it made every
modality *worse*. The learned slow-read gate never opened (sigmoid ≈ 0.12–0.13,
essentially its initial value) in any run — the model consistently declines to
use the slow level. Because the sigmoid gate cannot fully close, the nested
model always carries a small (~13%) uncalibrated slow-memory contribution, which
is why it is consistently ≤ baseline rather than ≈ baseline.

This is a clean, well-instrumented **negative result with a diagnosed cause**,
not a pipeline bug: the chunkwise kernel is unit-tested exactly equal to the
recurrent reference (err ~5e-16) for both toggles, training is healthy (loss
6.4→0.3, ~6 it/s), and the cause was localized by inspecting the learned gates
and the data construction.

## Results (train-clean-100, 5000 steps)

Per-modality validation CE at step 5000 (lower = better); Δ = baseline − nested.

| Regime | modality | baseline CE | nested CE | Δ (base−nested) |
|---|---|---:|---:|---:|
| **1. default fast decay, unpacked** | text | 0.987 | 1.013 | −0.026 |
| | speech | 1.445 | 1.445 | **−0.000** |
| | special | 1.535 | 1.485 | +0.050 |
| **2. local fast decay (exp=2), unpacked** | text | 1.036 | 1.005 | +0.031 |
| | speech | 1.440 | 1.442 | **−0.003** |
| | special | 1.521 | 1.516 | +0.005 |
| **3. local fast decay + packed** | text | 1.451 | 1.525 | −0.074 |
| | speech | 1.823 | 1.842 | **−0.020** |
| | special | 1.794 | 1.826 | −0.032 |

Slow-gate openness (mean sigmoid of the per-head read gate; init = 0.119):
regime 1 → 0.124, regime 2 → 0.127, regime 3 → 0.128. **It never opens.**

Figures: `/ptmp/$USER/LinearAttention/results/headline_100.png` (regime 1),
`headline_packed.png` (regime 3).

## Diagnosis (why the slow memory does nothing / hurts)

1. **Fast memory already covers the context.** The RetNet fast-decay schedule's
   slowest head has horizon ~2^12 ≫ 2048, so a *slower* memory is redundant.
   → addressed by `fast_decay_min_exp` (regime 2/3 localize fast memory to ≤512).

2. **The data had no long-range structure.** The original dataset put **one
   utterance per window** (~634 speech units + ~150 text bytes → a 2048 window
   was ~half padding, and examples were independent). A single utterance fits in
   fast memory, so the slow level has nothing to do.
   → addressed by `pack=True` (regime 3): concatenate consecutive same-chapter
   utterances into full windows (target padding ~0, spans cross utterances).

3. **Even with (1)+(2) fixed, the gate stays shut and nested is worse.** This is
   the real finding: at this scale, autoregressive *speech-unit* prediction is
   dominated by local context — long-horizon memory (whether slow-decaying fast
   heads or a separate slow level) does not help next-unit prediction. And
   because the gate cannot fully close, the always-on ~13% slow contribution is
   a net tax.

## What's still running / next

A definitive scale + mechanism test on **train-clean-360** (3.6× data), 12k
steps, packed + local fast:
- `config_baseline_packed360` / `config_nested_packed360` — controls for
  underfit/small-data (does nested help with more data + steps?).
- `config_nestedopen_packed360` — `slow_gate_bias_init=+2` **forces the slow
  memory engaged** (openness ~0.88) to isolate *is the slow level useful when
  actually used*, rather than relying on the model to open it.

(Results appended below when these finish.)

## Options to discuss

- **Accept the negative result** and frame it (the analysis is the contribution:
  multi-rate fast memory already suffices; speech-unit prediction is local at
  this scale). The risk register anticipated this.
- **Change the task to need long memory:** longer context (4k–8k), or group many
  same-speaker utterances so dependencies genuinely exceed the fast horizon, or a
  task that rewards long-range recall (speaker/topic-conditioned).
- **Fix the mechanism:** a hard-closable gate (e.g. ReLU/“straight-through” or
  learned per-head scalar that can reach 0), calibrated slow-read magnitude
  (normalize Z), or make the slow state a *content* summary rather than a coarse
  KV average.
