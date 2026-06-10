"""Speech-continuation discrimination probe: does long context help, and does
the nested slow memory help MORE as context grows?

Built entirely from LibriSpeech dev-clean (no external benchmark needed). For a
test point at an utterance boundary inside a chapter stream:

  context     = the preceding packed interleaved stream, truncated to the last
                N tokens (N swept over --context-lens)
  true cand.  = the actual next utterance's speech span  (<speech> u u ...)
  distractor  = a random other-SPEAKER utterance's speech span, fixed per point

The model scores mean log-prob of each candidate's unit tokens given the
context; accuracy = fraction of points where the true continuation wins.
N=0 gives the no-context floor (both candidates are plausible speech, so
~chance). If the nested slow memory is a real long-range speech memory, the
nested-minus-baseline accuracy gap should GROW with N -- specifically beyond
the fast-memory horizon (~512 tokens for fast_decay_min_exp=2).

    python -m nestedla.probe_continuation nestedla/config_nested_packed360.yaml \
        --ckpt .../ckpt-12000.pth
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from speechtext.corpus import UnitStore
from speechtext.ds import build_sequence
from speechtext.vocab import Vocab
from .config import TrainConfig
from .model import NestedLA, ModelArgs


def build_chapter_streams(units_dir: str, split: str, vocab: Vocab, seed: int):
    """Per-chapter packed interleaved streams + utterance boundary positions.

    Returns {chapter: {"toks": np.ndarray, "bounds": [(end_pos, utt_idx), ...]}}
    where end_pos is the stream position right AFTER an utterance's <eos> and
    utt_idx indexes the *next* utterance (the true continuation) in `index`.
    Also returns (units, index) for rendering candidates.
    """
    units, index, _ = UnitStore(units_dir).load(split)
    chapters = defaultdict(list)
    for i, rec in enumerate(index):
        spk, chap, _ = rec["utt_id"].split("-")
        chapters[f"{spk}-{chap}"].append(i)

    streams = {}
    for chap, idxs in chapters.items():
        toks_all, bounds = [], []
        for j, i in enumerate(idxs):
            rec = index[i]
            u = np.asarray(units[rec["start"]:rec["start"] + rec["length"]])
            rng = np.random.default_rng(seed + i)
            t, _ = build_sequence(vocab, rec["text"], u, bool(rng.integers(2)))
            toks_all.extend(t)
            if j + 1 < len(idxs):                      # next utt exists
                bounds.append((len(toks_all), idxs[j + 1]))
        streams[chap] = {"toks": np.asarray(toks_all, dtype=np.int64),
                         "bounds": bounds}
    return streams, units, index


def speech_span(units_arr, index, i, vocab, cand_units: int):
    rec = index[i]
    u = np.asarray(units_arr[rec["start"]:rec["start"] + rec["length"]])[:cand_units]
    return [vocab.speech] + vocab.encode_units(u)


@torch.no_grad()
def score_batch(model, items, vocab, device, autocast_ctx, chunk, batch_size):
    """items: list of (ctx_tokens, cand_tokens). Returns mean log-prob over each
    candidate's tokens (the <speech> tag onward, predicting units)."""
    out = np.zeros(len(items))
    order = np.argsort([len(c) + len(k) for c, k in items])
    for b0 in range(0, len(order), batch_size):
        ids = order[b0:b0 + batch_size]
        seqs = [np.concatenate([items[i][0], items[i][1]]) for i in ids]
        starts = [len(items[i][0]) for i in ids]        # candidate start in seq
        L = max(len(s) for s in seqs) - 1
        L = max(chunk, ((L + chunk - 1) // chunk) * chunk)
        x = torch.full((len(ids), L), vocab.pad, dtype=torch.long)
        y = torch.full((len(ids), L), vocab.pad, dtype=torch.long)
        cmask = torch.zeros((len(ids), L), dtype=torch.bool)
        for r, (s, st) in enumerate(zip(seqs, starts)):
            t = torch.from_numpy(s)
            x[r, : len(s) - 1] = t[:-1]
            y[r, : len(s) - 1] = t[1:]
            # targets that ARE candidate tokens: positions st-1 .. len(s)-2
            # (predicting s[st..]; s[st] is the <speech> tag's successor since
            #  the tag itself is given, we score units only)
            cmask[r, st: len(s) - 1] = True
        x, y = x.to(device), y.to(device)
        cmask = cmask.to(device)
        with autocast_ctx:
            logits = model(x)
            B, T, V = logits.shape
            ce = nn.functional.cross_entropy(
                logits.reshape(B * T, V), y.reshape(B * T),
                ignore_index=vocab.pad, reduction="none").reshape(B, T)
        lp = -(ce * cmask).sum(dim=1) / cmask.sum(dim=1).clamp(min=1)
        for r, i in enumerate(ids):
            out[i] = lp[r].item()
    return out


def main():
    ap = argparse.ArgumentParser(description="Speech-continuation discrimination probe")
    ap.add_argument("config")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--context-lens", type=int, nargs="+",
                    default=[0, 256, 512, 1024, 1792])
    ap.add_argument("--cand-units", type=int, default=200,
                    help="candidate speech units scored per continuation")
    ap.add_argument("--max-points", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
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
    print(f"[contin] loaded {args.ckpt} (step {ckpt.get('step')}), "
          f"slow_memory={model_args.slow_memory}")

    vocab = Vocab(n_units=model_args.vocab_size - 256 - 4)
    streams, units_arr, index = build_chapter_streams(
        config.units_dir, args.split, vocab, config.seed + 7)

    # test points: boundaries with enough preceding context, spread over chapters
    max_ctx = max(args.context_lens)
    points = []                       # (chapter, boundary_pos, true_utt_idx)
    for chap, st in streams.items():
        for pos, nxt in st["bounds"]:
            if pos >= max_ctx:
                points.append((chap, pos, nxt))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(points)
    points = points[: args.max_points]
    print(f"[contin] {len(points)} test points "
          f"({len(streams)} chapters), context lens {args.context_lens}")

    # fixed distractor per point: a different-speaker utterance
    speakers = [index[nxt]["utt_id"].split("-")[0] for _, _, nxt in points]
    distractors = []
    for k, (chap, pos, nxt) in enumerate(points):
        while True:
            j = int(rng.integers(len(index)))
            if index[j]["utt_id"].split("-")[0] != speakers[k]:
                distractors.append(j)
                break

    amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if device.type == "cuda" and amp_dtype is not torch.float32
                    else torch.autocast(device_type="cpu", enabled=False))

    res = {"experiment": config.experiment_name, "split": args.split,
           "slow_memory": bool(model_args.slow_memory), "step": ckpt.get("step"),
           "n_points": len(points), "cand_units": args.cand_units, "by_ctx": {}}
    for N in args.context_lens:
        items = []
        for k, (chap, pos, nxt) in enumerate(points):
            ctx = streams[chap]["toks"][max(0, pos - N): pos]
            true_c = np.asarray(speech_span(units_arr, index, nxt, vocab,
                                            args.cand_units), dtype=np.int64)
            dist_c = np.asarray(speech_span(units_arr, index, distractors[k],
                                            vocab, args.cand_units), dtype=np.int64)
            items.append((ctx, true_c))
            items.append((ctx, dist_c))
        lp = score_batch(model, items, vocab, device, autocast_ctx,
                         model_args.chunk_size, args.batch_size)
        lp_true, lp_dist = lp[0::2], lp[1::2]
        acc = float(np.mean(lp_true > lp_dist))
        margin = float(np.mean(lp_true - lp_dist))
        res["by_ctx"][str(N)] = {"acc": round(acc, 4), "margin": round(margin, 5)}
        print(f"[contin] ctx={N:5d}  acc={acc:.4f}  mean lp margin={margin:+.4f}")

    out = Path(args.out) if args.out else config.exp_dir / f"probe-continuation-{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[contin] wrote {out}")


if __name__ == "__main__":
    main()
