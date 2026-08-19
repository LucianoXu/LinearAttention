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

### Results (train-clean-360, 12000 steps, packed + local fast decay)

Per-modality validation CE at step 12000; Δ = baseline − nested (>0 ⇒ nested better):

| modality | baseline CE | nested CE | nested-open CE | Δ (nested) | Δ (nested-open) |
|---|---:|---:|---:|---:|---:|
| text | 0.900 | 0.903 | 0.907 | −0.003 | −0.008 |
| **speech** | 1.242 | **1.231** | 1.247 | **+0.011** | −0.006 |
| special | 1.190 | 1.203 | 1.208 | −0.013 | −0.018 |

As perplexity (nested vs baseline): **speech 3.461 → 3.423 (−1.08%)**, text
2.459 → 2.466 (+0.27%, flat), special 3.288 → 3.331. Figure:
`results/headline_packed360.png`.

**This is the hypothesised asymmetry, and it EMERGED with scale.** At
train-clean-100/5k steps the slow memory was flat-to-harmful; at
train-clean-360/12k steps the nested model lowers **speech** PPL by ~1% while
**text** is unchanged — Δspeech > 0 and Δspeech > Δtext, exactly the predicted
direction. The effect is small and single-seed, but it trends the right way with
data + steps.

**Gate behaviour confirms the mechanism wants a *light touch*:**
- learned nested gate openness rose from 0.128 (100/5k) → **0.138** (360/12k),
  with the gate weights growing (0.02 → 0.027, i.e. it became input-dependent) —
  a small, learned engagement is what produced the speech gain.
- `nested-open` (gate **forced** to 0.85 via `slow_gate_bias_init=+2`) is **worse
  than baseline on every modality** (speech −0.006). Forcing the slow memory
  fully on hurts; the model's optimal use is a small dose.

So the slow level is *useful but only in small, learned amounts*, and its benefit
is *speech-specific* and *scale-emergent*. A natural next test is longer context
(below).

### Context-4096 (train-clean-360, 8000 steps, packed + local fast)

PPL (nested vs baseline): speech 3.372 → 3.366 (Δ +0.17%), text 2.274 → 2.276
(flat), special 3.008 → 2.972 (+1.17%). Δspeech = +0.006 > Δtext = −0.002 — same
direction (nested helps speech, text flat), gate openness 0.131. Figure:
`results/headline_ctx4k360.png`.

**Caveat — not a clean scaling comparison.** To fit 4096 context this run used
batch 4 / 8000 steps = ~524M training tokens, vs ~786M for the 2048 run, so it
is *under-trained* relative to ctx-2048; the smaller Δspeech cannot be attributed
to context length. The direction is consistent. A **budget-matched context
sweep** (equal tokens at 2048 vs 4096 vs 8192) is the right way to test whether
the speech gain grows with context — left as the first follow-up.

## Summary of all runs

| # | data / steps | context | fast decay | packed | Δspeech (PPL) | Δtext (PPL) | gate | verdict |
|---|---|---|---|---|---:|---:|---:|---|
| 1 | 100 / 5k | 2048 | default | no | ~0 | −0.07 | 0.124 | flat |
| 2 | 100 / 5k | 2048 | local | no | ~0 | +0.07 | 0.127 | flat |
| 3 | 100 / 5k | 2048 | local | yes | −0.12 | −0.33 | 0.128 | nested worse |
| 4 | **360 / 12k** | 2048 | local | yes | **+0.038 (−1.08%)** | −0.007 (flat) | 0.138 | **speech-specific gain** |
| 4b | 360 / 12k | 2048 | local | yes, gate forced open | −0.02 | −0.02 | 0.851 | forcing hurts |
| 5 | 360 / 8k | 4096 | local | yes | +0.006 | −0.002 | 0.131 | consistent, under-trained |

(Δ = baseline − nested PPL; positive ⇒ nested better. Run 4 is the headline.)

## Day 5 (2026-06-10 evening): scale-up to 330M / 960h / ctx-8192 + an architecture fix

Two changes this round, then the headline A/B re-run at ~3.5× params, 2.7× data,
4× context:

**Architecture fix (`nestedla/model.py`):**
1. **RoPE off by default** (`use_rope=False`). The per-head decay already encodes
   position (RetNet-style), so RoPE is redundant on the fast path *and* corrupts
   the slow read: chunk-averaging rotated keys cancels high-freq phase, and the
   huge relative phase over thousands of tokens turns `q·k_slow` into noise.
2. **fp32 decay tables + cross-chunk state.** In bf16 the slow heads' gammas
   (close to 1) round to *exactly* 1.0, silently removing their decay so the
   recurrent state grows unbounded. Compute now promotes to fp32; the fp64
   equivalence test still passes (chunkwise == reference, err ~1e-9, both toggles).

**Run:** 24L d=1024 (329M), ctx 8192, chunk 256, packed, `fast_decay_min_exp=2`,
train-960 (281k utts / 173M units = 100+360+500), 18k steps, 4×A100, ~1 s/step
(jobs 27815346/47). Eval at the **best checkpoint, step 10500** (see overtraining
note), full dev-clean, job 27829662. Figures in `results/960L8k/`.

### Overtraining (operational finding)

Dev speech-CE bottomed at **step ~10200** for both models, then rose steadily to
step 18000 (baseline speech CE 1.065 → 1.139, +7%); 18k steps ≈ 14 epochs of
173M units was too many with cosine decay to ~0 LR. **The final checkpoint is not
the best one** — all numbers below use step 10500. Next runs should cap ~10k
steps or add early stopping.

### The architecture fix worked — and that makes the result a *stronger* negative

| metric (step 10500, full dev-clean) | baseline | nested | Δ |
|---|---:|---:|---:|
| speech PPL | **2.959** | 2.977 | **−0.6% (nested worse)** |
| text PPL | 1.945 | 1.936 | +0.4% (nested better) |
| slow-gate openness (mean) | — | **0.209** | (init 0.119; was 0.138 at 93M) |

**The gate opened — and the model genuinely uses the slow level now.** Per-head
openness: `0.012 0.019 0.031 0.041 0.138 0.222 0.414 0.797` — the four
fastest-decay heads stay ~shut, but the slowest head opens to **0.80** and the
next two to 0.41 / 0.22. So the fix did exactly what it should mechanically: the
slow gammas no longer collapse to 1.0 and RoPE no longer corrupts the read, so
the slow memory is actually engaged where a slow memory makes sense (the slow
heads). **And yet speech PPL is slightly *worse*.**

This kills the old escape hatch. At 93M the easy story was "the model declines
to open the gate." Now it opens it, engages the slow heads, and the slow memory
*still does not help* speech-unit prediction — it slightly hurts. The 93M
positive (+1.08% speech, gain growing with context) **did not survive scale + the
fix**: the distance probe now shows a small *uniform* speech deficit at every
window position (Δspeech −0.005 to −0.008 beyond 256 tokens), not a
context-growing gain. The continuation probe agrees — nested is ≤ baseline at
every context length (e.g. 0.748 vs 0.773 at ctx 2048), converging only at the
very longest context (0.765 vs 0.773 at 7936). The 93M effect is best read as a
small-model/old-bug artifact, not a robust mechanism.

### Why: LibriSpeech has no exploitable structure past ~40 s (the corpus is the wall)

The continuation probe at 330M / ctx 8192 is the cleanest evidence yet:

| ctx tokens | 0 | 512 | 1024 | 2048 | 4096 | 7936 |
|---|---:|---:|---:|---:|---:|---:|
| baseline acc | 0.530 | 0.722 | 0.760 | **0.773** | 0.775 | 0.773 |

Accuracy climbs steeply to ~0.77 by **2048 tokens (~40 s)** and is then **flat**
(4096, 7936 add nothing) — a 330M model with 8192-token context extracts no
additional predictive signal from read-audiobook speech beyond ~40 s. There is
nothing for a slow memory to carry, so engaging it can only add variance. Spoken
StoryCloze stays at chance for both models (sSC 0.47–0.48, tSC 0.49; per-token
tSC 0.52–0.53), confirming the benchmark is scale-gated well past 330M.

### Day 5 verdict

The mechanism is no longer in question and the data is: with the slow level
genuinely engaged (gate 0.80 on the slow head), it does not help — slightly
hurts — next-unit prediction at 330M / 960h, and the predictable structure in
LibriSpeech saturates by ~40 s. **The bottleneck is the corpus, not the model or
its willingness to use slow memory.** This is the decisive motivation to move to
a corpus with genuine long-range structure (conversational / podcast, e.g.
GigaSpeech), where the continuation signal would *not* plateau at 2 k tokens —
that is the next experiment.

## Day 5 addendum: RoPE × nested ablation at 93M/360h (job 27817337, recorded 2026-08-19)

Ran on 2026-06-10 (17:45) but not written up until now. A 2×2 at the original
93M/360h packed scale — {arch-fix i.e. RoPE off, RoPE on} × {baseline, nested} —
with early stopping: all four evaluated at **step 9000** (before the overtraining
regime seen at 330M). Configs: `config_*_p360fix_es.yaml` /
`config_*_p360rope_es.yaml`. Full dev-clean PPL:

| config (93M/360h @ step 9000) | speech PPL | text PPL | special PPL | gate |
|---|---:|---:|---:|---:|
| fix (RoPE off) baseline | 3.419 | 2.348 | 3.256 | — |
| fix (RoPE off) nested | 3.472 (**+1.5% worse**) | 2.319 | 3.301 | 0.223 |
| rope baseline | 3.376 | 2.349 | 3.349 | — |
| rope nested | **3.281 (−2.8% better)** | 2.305 (−1.9%) | 3.287 | 0.229 |

Distance probe (window-position speech Δ = baseline − nested CE, >0 ⇒ nested
better): **rope** grows monotonically +0.002 → +0.009 → +0.024 → +0.031 →
**+0.034** across buckets — the context-growing signature, cleaner and ~3×
larger than the original packed360 headline. **fix** is negative in every bucket
(−0.003 → −0.017): nested uniformly worse. Gate openness is essentially
identical in both nested runs (0.22–0.23, slowest heads 0.6–0.8), so the fp32
decay fix does open the gate at 93M too — engagement is not the differentiator.

**This complicates the Day 5 story.** The two changes between the old 93M
positive and the 330M negative were (a) scale and (b) the arch fix (RoPE off +
fp32 decay). This ablation isolates (b) at fixed scale: *turning RoPE off is
itself what flips nested from helping (−2.8% speech, context-growing) to
hurting (+1.5%)*. That is the opposite of the Day 5 theoretical diagnosis
("RoPE corrupts the slow read") — empirically the slow memory only pays off
*with* RoPE on the fast path. Two readings:

1. The 330M negative may be partly an artifact of removing RoPE, not scale —
   an un-run `rope` A/B at 330M/8192 would settle it. (Caveat: the RoPE-phase
   problem grows with context, so RoPE-on at 8192 may genuinely break.)
2. Alternatively, with RoPE off the *fast* heads get better at long range
   (nothing scrambles their phases), shrinking the slow level's niche — i.e.
   the nested gain at 93M was compensating for RoPE-induced fast-path damage
   rather than adding new capability. Consistent with rope-baseline ≈
   fix-baseline on speech (3.376 vs 3.419, RoPE slightly *better*).

Either way, "the 93M effect was a small-model/old-bug artifact" (Day 5 verdict)
is too strong: at matched scale and matched early-stopped checkpoints the
effect reproduces with RoPE on, and the arch fix — not scale — removes it. The
corpus-saturation finding (continuation probe flat past ~40 s) stands
regardless and remains the reason to move to conversational data.

## Day 4 (2026-06-10): benchmarks + mechanism probes

Three new evaluations on the headline packed360 pair (baseline vs nested,
ckpt-12000), all single-GPU (job 27810343; code: `nestedla/probe_distance.py`,
`nestedla/probe_continuation.py`, `nestedla/eval_storycloze.py` +
`speechtext/prepare_storycloze.py`, comparison via `nestedla/plot_probes.py`).

### 1. Where the speech gain lives: distance-bucketed CE (the mechanism probe)

Per-modality CE on packed dev-clean, bucketed by window position (= context
available to the prediction). Δ = baseline − nested CE (>0 ⇒ nested better):

| bucket | Δ speech | n speech | Δ text | n text |
|---|---:|---:|---:|---:|
| [0,128) | +0.0070 | 60k | +0.0055 | 17k |
| [128,256) | +0.0084 | 60k | +0.0001 | 18k |
| [256,512) | +0.0090 | 121k | −0.0076 | 35k |
| [512,1024) | +0.0115 | 239k | −0.0011 | 74k |
| [1024,2048) | **+0.0122** | 485k | −0.0026 | 142k |

**The nested speech gain grows monotonically with available context** (+0.007 →
+0.012, ~1.7×) while the text Δ is flat-to-negative — exactly the signature the
slow-memory mechanism predicts: the aggregate −1.08% speech PPL is concentrated
beyond the fast-memory horizon (~512), where only the slow level can carry
information. Bucketing by tokens-since-utterance-start shows the same growth
(+0.008 at [0,128) → +0.024 at [1024,2048)), i.e. the gain also deepens within
long utterances; it is context-volume, not specifically cross-utterance recall.
The under-trained ctx-4096 pair shows the same direction but ~6× weaker
(Δspeech ≈ +0.002 at long range), consistent with its under-training.
Figure: `results/probe_distance.png`.

### 2. Continuation discrimination: long context helps, nested doesn't help more

400 utterance boundaries in dev-clean chapter streams; score the true next
utterance's speech span vs a random other-speaker span (mean log-prob given the
preceding N tokens). Accuracy (chance 0.5):

| ctx tokens | baseline | nested | Δ |
|---|---:|---:|---:|
| 0 | 0.515 | 0.493 | −0.023 |
| 256 | 0.625 | 0.615 | −0.010 |
| 512 | 0.665 | 0.645 | −0.020 |
| 1024 | 0.690 | 0.685 | −0.005 |
| 1792 | 0.693 | 0.688 | −0.005 |

The probe itself is healthy — accuracy climbs 51%→69% with context, and keeps
inching up past the fast horizon. But **nested ≈ baseline at every context
length** (differences within noise at n=400, SE ≈ 2.3%): the slow memory's PPL
gain does not convert into better long-range *discrimination*.
Figure: `results/probe_continuation.png`.

### 3. Spoken StoryCloze (sSC / tSC, slprl multispeaker, bm voice, n=1871)

Standard SpeechLM benchmark: correct- vs wrong-ending spoken stories, accuracy
by total log-likelihood (per-token in parens):

| task | baseline | nested |
|---|---:|---:|
| sSC | 0.486 (0.517) | 0.483 (0.507) |
| tSC | 0.486 (0.512) | 0.488 (0.507) |

**Both models are at chance.** Expected at this scale: TWIST-style models need
~1B+ params and orders more data to clear 55–80% here, and Kokoro-TTS audio is
a domain shift from LibriSpeech audiobook speech (HuBERT units still transfer:
per-token accuracies are marginally >0.5). The benchmark has no headroom to
separate the pair at 93M/360h — a scale gate, not a negative for the mechanism.

### Day 4 verdict

The distance probe upgrades the headline result from "small aggregate PPL gain"
to **a gain that lives where a slow memory should act and grows with context** —
the strongest mechanistic evidence so far. But two zero-shot *discriminative*
tests (continuation, StoryCloze) show no benefit: the slow level improves dense
next-unit prediction, not yet usable long-range decisions. That gap (PPL ≠
ability) is the honest framing for the showcase, and it sharpens the next
steps: scale (data/params), a hard-closable gate, and long-form data where
decisions *require* memory.

## Conclusion

The nested two-level memory delivers a **small, speech-specific perplexity gain
that emerges with scale** (run 4: −1.08% speech PPL, text unchanged), in the
direction the Nested-Learning hypothesis predicts. It is **not** a large effect
and it requires the right regime — enough data/steps, packed long-context
windows, and fast memory localized below the context length so the slow level
has a distinct job. The slow memory must be used in *small, learned doses*
(forcing the gate open hurts). Whether the effect scales further (more data,
longer context, budget-matched) is the open question for discussion.

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
