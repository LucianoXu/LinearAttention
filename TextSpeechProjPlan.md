# Nested Multi-Timescale Linear-Attention for Joint Speech-Text LM

**Raven-adapted execution plan.** One-week showcase project for intern/postdoc
applications. The design (the *what*) was fixed during brainstorming; this
document re-scopes the *how* to the Raven (MPCDF) cluster and the existing
`LinearAttention` repo.

---

## 0. TL;DR — what changed for Raven

The original brainstorm assumed a generic dev box with `pip`, the
`flash-linear-attention` (fla) library, `torchaudio`, and `wandb`. Raven is an
HPC cluster with no root, no long interactive jobs, no compute-node internet,
and a fixed PyTorch **container**. Concrete adaptations:

| Original plan | Raven-adapted plan | Why |
|---|---|---|
| `fla` GLA blocks (Triton kernels) | **Chunkwise-parallel fast path in pure PyTorch**, built on the repo's RetNet decay primitive; `fla` is an opt-in, benchmarked bonus | The repo's current RetNet trains via the **O(L²)** parallel form (`retnet/model.py:196-203` materializes a `(B,H,L,L)` matrix — no linear-attention efficiency at train time). The big win is the **algorithm** (O(L²)→O(L·C)), which is fla-independent; fla's triton 2.2 / torch 2.3a compat is day-1 risk and it has no slow-memory anyway |
| `wandb` | **tensorboard** (already the repo convention, bundled in container) | No compute-node internet; matches `transformer/`, `retnet/` |
| `torchaudio` HuBERT pipeline | **`transformers` HubertModel** + `librosa`/`soundfile` (already in container) | torchaudio is missing & version-matching torch 2.3a is fiddly; only one new dep (`transformers`) |
| `pip install` env | `image_pytorch/2024.02` **container** + `pip --user` *through* the container on the login node | DDP needs the container (stock modules lack `torch.distributed`); `~/.local/python3.10` matches |
| Download data at runtime | **Stage everything to `/ptmp` on the login node** first | Compute nodes have no internet |
| "small experiments on a GPU" | `gpudev` (≤15 min smoke) + `gpu` (≤24 h runs); **multi-node = parallel independent sweep jobs**, not multi-node DDP | This is how Raven schedules; the ablation is embarrassingly parallel |

Everything lands on **scratch**: `/ptmp/$USER/LinearAttention/{ds,units,ckpt,logs,.hf,.inductor-cache}`.

---

## 1. Core hypothesis (unchanged — the thing the experiment tests)

Speech-unit spans are long and dense (≈50 tokens/sec); text is compact. A single
fast-decaying linear-attention memory under-serves long speech spans. Adding a
slower, lower-frequency memory level supplies cheap long-horizon context → the
perplexity gain should be **larger on speech tokens than text tokens**. That
asymmetric Δ is the headline result and the empirical embodiment of Nested
Learning's multi-frequency memory.

## 2. Data & tokenization

- **Speech tokenizer (frozen):** `facebook/hubert-base-ls960` via
  `transformers.HubertModel`, layer-9 features → k-means (K=500, sklearn, already
  in container), fit on ~1–2 h of audio. ~50 Hz, single unit stream, **no dedup**
  (preserve density contrast). Units cached to `/ptmp/.../units/`.
- **Text:** byte-level (256 tokens) — robust, zero training.
- **Combined vocab (~770):** bytes + units + specials (`<text>`, `<speech>`,
  `<eos>`, `<pad>`). Single embedding + single softmax.
- **Data:** LibriSpeech `train-clean-100` (subset for fast iteration),
  `dev-clean` for eval. Staged to `/ptmp` on the login node (wget the OpenSLR
  tarballs; HF weights cached under `HF_HOME=/ptmp/$USER/.hf`).
- **Sequences:** utterance-level interleaving, random direction per example —
  `<text>…<speech>…` and `<speech>…<text>…`. No forced alignment. Context ~2048.

Shared data code lives in a new **`speechtext/`** package (used by both baseline
and nested models): `tokenizer.py` (HuBERT→kmeans), `prepare_units.py` (one-shot
caching script), `ds.py` (interleaved dataset + DDP-aware sharding).

## 3. Model

New portable package **`nestedla/`**, built on the repo's RetNet primitive,
following the existing package layout (`model.py`, `config.py`, `config.yaml`,
`ds.py`→imports `speechtext.ds`, `train.py`, `generate.py`, `test_nestedla.py`):

- Decoder-only LM, RetNet-style decay-linear-attention blocks + MLP + RMSNorm.
  ~100M (d=768, 12 layers), configurable; fits comfortably on one 40 GB A100.
- **Fast memory — chunkwise-parallel** (the key implementation change): process
  the sequence in chunks of size `C`, carry the recurrent KV state across chunk
  boundaries, compute intra-chunk contributions in parallel. This is
  O(L·C·d) compute / memory instead of the current O(L²) parallel form, never
  materializes the `(B,H,L,L)` matrix, and is `torch.compile`-friendly. Decay is
  multi-rate across heads (geometric range), as in the existing RetNet.
  *Equivalence to the recurrent form is unit-tested* (like the existing
  `test_*` parallel/recurrent checks).
- **Slow memory (the contribution):** a second associative state updated **at the
  chunk boundaries** (every `K`, aligned to chunk size) with slow decay; each
  token reads fast+slow via a learned gate. The chunkwise structure is what makes
  this natural — the boundary between chunks *is* the per-K slow-update point.
  Update frequencies literally differ (per-token fast vs per-K slow).
- **Baseline (flat):** identical package, `slow_memory: false` in the config —
  the A/B is one config toggle. Slow level adds a few params; report the small
  overhead.

`train.py` copies `transformer/train.py`'s DDP treatment (torchrun, bf16
autocast, TF32, fused AdamW, sharded validation, rank-0 logging, ckpt to
`/ptmp`). Add `"nestedla"` to `raven-specific/train_main.py`'s `--model`
choices and add `raven-specific/slurm/train_nestedla.sbatch`.

## 4. Evaluation

- Per-modality next-token CE/PPL (text / speech / cross-modal-conditional),
  masked by token type.
- **Headline plot:** baseline vs nested, per modality → expect Δspeech ≫ Δtext.
- Secondary ablations: chunk size `K`, number of timescales.
- **Training-performance result (secondary showcase point):** throughput
  (tokens/s) and peak GPU memory vs context length for **naive O(L²) → chunkwise
  PyTorch → fla** (when fla installs cleanly). A clean speed/memory curve is a
  strong systems-side thing to show a reviewer, alongside the modeling ablation.
- Stretch (not required for the bar): sample speech units from text, decode with
  a unit HiFi-GAN vocoder for listenable demos.

## 5. Raven execution mechanics (read once, reuse everywhere)

**One-time setup on the login node** (`raven01`, has internet):

```bash
module load image_pytorch/2024.02            # provides $IMAGE_SIF + apptainer
# extra wheels into ~/.local/python3.10 (matches the container python):
apptainer exec "$IMAGE_SIF" pip install --user transformers
# (sklearn, librosa, soundfile, tensorboard are ALREADY in the container)

# stage data + model weights to scratch (no internet on compute nodes):
mkdir -p /ptmp/$USER/LinearAttention/{ds,units,ckpt,logs,.hf}
export HF_HOME=/ptmp/$USER/.hf
#  - LibriSpeech tarballs -> /ptmp/.../ds   (wget from OpenSLR, then untar)
#  - HuBERT weights:  apptainer exec $IMAGE_SIF python -c "from transformers import HubertModel; HubertModel.from_pretrained('facebook/hubert-base-ls960')"
```

**Smoke test (≤15 min, `gpudev`)** — always do this before a real run:

```bash
sbatch --partition=gpudev --time=00:15:00 \
  raven-specific/slurm/train_nestedla.sbatch nestedla/config_smoke.yaml
```

**Real run (`gpu`, 1 node = 4×A100, ≤24 h):**

```bash
sbatch raven-specific/slurm/train_nestedla.sbatch nestedla/config_baseline.yaml
sbatch raven-specific/slurm/train_nestedla.sbatch nestedla/config_nested.yaml
```

**Using multiple nodes = parallel independent runs** (the ablation is
embarrassingly parallel; one node per config). Either submit configs separately
or use a SLURM job array, e.g. a sweep over `{baseline, nested, K∈{…}, timescales∈{…}}`
each as its own 1-node job. This is far simpler and more robust than multi-node
NCCL DDP and matches how the ablation is structured.

**Caches/paths** (off the home quota): `HF_HOME`, `TORCHINDUCTOR_CACHE_DIR`,
units, ckpts, logs all under `/ptmp/$USER/LinearAttention/...` — same as the
existing `train_transformer.sbatch`.

## 6. Agent-driven workstreams (parallelizable)

1. Env + staging: `pip --user transformers`, stage LibriSpeech + HuBERT to `/ptmp`.
2. `speechtext/tokenizer.py` + `prepare_units.py` (HuBERT→kmeans→cached units) ∥ run on `gpudev`.
3. `speechtext/ds.py` (interleaved tokenized sequences, DDP sharding) + a tiny correctness test.
4. `nestedla/` model: **chunkwise-parallel fast memory** + slow memory + toggle + `test_nestedla.py` (chunkwise vs recurrent equivalence, like the existing tests).
5. `nestedla/train.py` (DDP, per-modality loss logging to tensorboard) + sbatch + `train_main.py` hookup.
6. Eval + per-modality plotting script.
7. **Perf benchmark script** (throughput + peak memory vs context: naive → chunkwise → fla).
8. (Stretch) fla fast-path swap; (stretch) unit vocoder.

Early streams (1–3) run as background work on `gpudev`/login while 4–6 are built.

## 7. Seven-day schedule (Raven turnaround baked in)

| Day | Goal | Gate |
|---|---|---|
| 1 | Env + staging (transformers, LibriSpeech, HuBERT to /ptmp); `speechtext` tokenizer; byte text tokenization; fit k-means | data→tokens works on `gpudev` |
| 2 | `speechtext/ds.py` interleaving + sharding; `nestedla` flat baseline with **chunkwise fast path** + chunkwise/recurrent equivalence test; overfit-a-batch smoke on `gpudev` | correctness check |
| 3 | Full baseline run on `gpu`; per-modality PPL eval | floor: end-to-end proof-of-life |
| 4 | Slow-memory module + toggle; recurrence/parallel equivalence test; overfit check; launch nested run | nested trains |
| 5 | Nested vs baseline complete; headline per-modality plot | **target met** |
| 6 | Parallel sweep across nodes (K, #timescales); perf benchmark (naive→chunkwise→fla); scale up if quota allows; stretch fla swap / vocoder | ablation + perf curves |
| 7 | README + figures + short report framing NL↔bimodal story; buffer | showcase ready |

## 8. Risk register (Raven-specific additions in **bold**)

- **`transformers` HuBERT load fails offline** → pre-cache weights to `HF_HOME`
  on the login node; set `HF_HUB_OFFLINE=1` on compute nodes.
- **Container torch 2.3a quirks** (e.g. Triton `libcuda` lookup) → already
  handled by `raven-specific/slurm/_in_container.sh`; reuse it.
- **`fla` install fails on triton 2.2 / torch 2.3a** → fla is a *bonus* only; the
  chunkwise-PyTorch fast path is the de-risked default and needs no fla. If fla
  won't pin cleanly, the benchmark just compares naive vs chunkwise.
- **Chunkwise ≠ recurrent (numerical bug)** → unit-test equivalence against the
  recurrent form before any training run (Day 2 gate), as done for the existing
  models.
- HuBERT+kmeans fiddly → fallback: single-codebook neural codec
  (EnCodec/WavTokenizer) — but adds a dependency, so prefer HuBERT.
- Nested shows no gain → fallback to decay-spread-only (nested reduces to it);
  lengthen context / raise speech density to make slow memory matter. An
  analyzed negative result is still presentable; we aim for signal.
- Time slip → flat RetNet baseline is the de-risked floor; self-modifying memory
  is explicitly future work.
- **`gpu` partition queue wait** → smoke on `gpudev` first; submit sweep jobs
  early and let them queue; checkpoints on `/ptmp` survive across jobs.
