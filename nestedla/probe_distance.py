"""Distance-bucketed per-modality CE probe (mechanism evidence).

The headline result (RESULTS.md) is an *aggregate* speech-PPL gain from the
nested slow memory. This probe asks WHERE in the context that gain lives, by
bucketing next-token CE two ways on packed windows:

  1. **window position** of the target token -- i.e. how much context is
     available. With local fast decay (horizon ~512 << context), the slow
     level is the only memory that can span the whole window; if the speech
     gain is really a long-range-memory effect it should concentrate in
     buckets *beyond* the fast horizon, not at the start of the window.

  2. **distance since the current utterance started** (tokens since the last
     <eos> in the input). Targets early in an utterance can only profit from
     *cross-utterance* context; targets deep inside an utterance are largely
     predictable from within it. A slow-memory gain at small utterance-
     distance but large window position = cross-utterance recall.
     (Positions before the first <eos> of a window get distance-from-window-
     start, a lower bound; with ~800-token utterances in 2048 windows this
     affects a minority of positions and both models equally.)

Run it once per checkpoint; compare JSONs across the baseline/nested pair:

    python -m nestedla.probe_distance nestedla/config_nested_packed360.yaml \
        --ckpt /ptmp/$USER/LinearAttention/ckpt/nestedla-nested-packed360/ckpt-12000.pth
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from speechtext.ds import SpokenLMDataset, make_collate, MOD_TEXT, MOD_SPEECH, MOD_SPECIAL
from speechtext.vocab import Vocab
from .config import TrainConfig
from .model import NestedLA, ModelArgs

_MODS = {"text": MOD_TEXT, "speech": MOD_SPEECH, "special": MOD_SPECIAL}


def make_edges(context_len: int) -> list[int]:
    """Power-of-2 bucket edges: [0,128,256,512,1024,...,context_len]."""
    edges = [0]
    e = 128
    while e < context_len:
        edges.append(e)
        e *= 2
    edges.append(context_len)
    return edges


def bucketize(pos: torch.Tensor, edges: list[int]) -> torch.Tensor:
    """Map positions to bucket indices given edges (len(edges)-1 buckets)."""
    b = torch.zeros_like(pos)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        b = torch.where((pos >= lo) & (pos < hi), torch.full_like(pos, i), b)
    return b


def utt_distance(x: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Per-position #tokens since the last <eos> in x (B,T). No eos yet =>
    distance from window start (lower bound on the true distance)."""
    B, T = x.shape
    idx = torch.arange(T, device=x.device).expand(B, T)
    last_eos = torch.where(x == eos_id, idx, torch.full_like(idx, -1))
    last_eos = torch.cummax(last_eos, dim=1).values
    return idx - last_eos - 1  # token right after <eos> has distance 0


@torch.no_grad()
def probe(model, loader, vocab, device, autocast_ctx, edges, max_batches=None):
    model.eval()
    NB = len(edges) - 1
    # sums/counts: [n_mods, n_buckets] for each bucketing
    sums = {k: torch.zeros(len(_MODS), NB, device=device) for k in ("pos", "utt")}
    cnts = {k: torch.zeros(len(_MODS), NB, device=device) for k in ("pos", "utt")}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["target_ids"].to(device, non_blocking=True)
        ymod = batch["target_mod"].to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(x)
            B, T, V = logits.shape
            ce = nn.functional.cross_entropy(
                logits.reshape(B * T, V), y.reshape(B * T),
                ignore_index=vocab.pad, reduction="none").reshape(B, T)
        pos = torch.arange(T, device=device).expand(B, T)
        buckets = {"pos": bucketize(pos, edges),
                   "utt": bucketize(utt_distance(x, vocab.eos).clamp(min=0), edges)}
        for mi, (_, code) in enumerate(_MODS.items()):
            m = (ymod == code)
            for k in sums:
                bk = buckets[k][m]
                sums[k][mi].scatter_add_(0, bk, ce[m])
                cnts[k][mi].scatter_add_(0, bk, torch.ones_like(ce[m]))
    out = {}
    for k in sums:
        ce_kb = (sums[k] / cnts[k].clamp(min=1)).cpu()
        out[k] = {"edges": edges}
        for mi, name in enumerate(_MODS):
            out[k][name] = {"ce": [round(v, 6) for v in ce_kb[mi].tolist()],
                            "n": [int(v) for v in cnts[k][mi].cpu().tolist()]}
    return out


def main():
    ap = argparse.ArgumentParser(description="Distance-bucketed per-modality CE probe")
    ap.add_argument("config", help="training YAML (for data/vocab settings)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = TrainConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_args = ckpt["args"] if isinstance(ckpt.get("args"), ModelArgs) else config.model_args
    model = NestedLA(model_args).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[probe] loaded {args.ckpt} (step {ckpt.get('step')}), "
          f"slow_memory={model_args.slow_memory}")

    vocab = Vocab(n_units=model_args.vocab_size - 256 - 4)
    # packing is the point of this probe: full windows, real long-range context
    ds = SpokenLMDataset(config.units_dir, args.split, vocab,
                         context_len=model_args.context_len, seed=config.seed + 7,
                         pack=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=config.num_workers, pin_memory=True,
                        collate_fn=make_collate(vocab, model_args.context_len),
                        drop_last=False)

    amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if device.type == "cuda" and amp_dtype is not torch.float32
                    else torch.autocast(device_type="cpu", enabled=False))

    edges = make_edges(model_args.context_len)
    res = probe(model, loader, vocab, device, autocast_ctx, edges, args.max_batches)
    res.update(experiment=config.experiment_name, split=args.split,
               slow_memory=bool(model_args.slow_memory), step=ckpt.get("step"))

    for k in ("pos", "utt"):
        lab = "window-pos" if k == "pos" else "utt-dist"
        for name in _MODS:
            cells = " ".join(f"[{lo}-{hi}) {c:.4f}" for lo, hi, c in
                             zip(edges[:-1], edges[1:], res[k][name]["ce"]))
            print(f"[probe] {lab:10s} {name:7s} {cells}")

    out = Path(args.out) if args.out else config.exp_dir / f"probe-distance-{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
