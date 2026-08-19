"""Standalone per-modality evaluation for a trained nested speech-text LM.

Loads a checkpoint and reports next-token cross-entropy / perplexity *split by
modality* (text bytes vs speech units vs special tags) over a whole validation
split. This is the rigorous version of the per-modality validation that
train.py already logs during training -- here we run a single GPU over the full
split (or --max-batches of it) and dump a JSON the plotting script can consume.

    python -m nestedla.eval nestedla/config_baseline.yaml \
        --ckpt /ptmp/$USER/LinearAttention/ckpt/nestedla-baseline/ckpt-5000.pth \
        --split dev-clean

The headline number is PPL_speech vs PPL_text; comparing baseline vs nested,
the nested slow memory should lower speech PPL more than text PPL.
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


@torch.no_grad()
def evaluate_split(model, loader, pad_id, device, autocast_ctx, max_batches=None):
    model.eval()
    sums = {k: torch.zeros((), device=device) for k in ("loss", *_MODS)}
    counts = {k: torch.zeros((), device=device) for k in ("loss", *_MODS)}
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
                ignore_index=pad_id, reduction="none").reshape(-1)
        fm = ymod.reshape(-1)
        valid = (fm != -1)
        sums["loss"] += ce[valid].sum(); counts["loss"] += valid.sum()
        for name, code in _MODS.items():
            m = (fm == code)
            sums[name] += ce[m].sum(); counts[name] += m.sum()
    ce_per = {k: (sums[k] / counts[k].clamp(min=1)).item() for k in sums}
    out = {}
    for k, v in ce_per.items():
        out[f"ce_{k}"] = v
        out[f"ppl_{k}"] = math.exp(v) if v < 50 else float("inf")
        out[f"tokens_{k}"] = int(counts[k].item())
    return out


def main():
    ap = argparse.ArgumentParser(description="Per-modality eval of a nestedla checkpoint")
    ap.add_argument("config", help="training YAML (for data/vocab settings)")
    ap.add_argument("--ckpt", required=True, help="path to ckpt-*.pth")
    ap.add_argument("--split", default=None, help="override valid_split (e.g. dev-clean)")
    ap.add_argument("--max-batches", type=int, default=None, help="cap #batches (default: full split)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None, help="output JSON path (default: <exp_dir>/eval-<split>.json)")
    args = ap.parse_args()

    config = TrainConfig.from_yaml(args.config)
    split = args.split or config.valid_split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_args = ckpt["args"] if isinstance(ckpt.get("args"), ModelArgs) else config.model_args
    model = NestedLA(model_args).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[eval] loaded {args.ckpt} (step {ckpt.get('step')}), "
          f"slow_memory={model_args.slow_memory}, params={sum(p.numel() for p in model.parameters())}")

    vocab = Vocab(n_units=model_args.vocab_size - 256 - 4)
    collate = make_collate(vocab, model_args.context_len)
    ds = SpokenLMDataset(config.units_dir, split, vocab,
                         context_len=model_args.context_len, seed=config.seed + 7)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=config.num_workers, pin_memory=True,
                        collate_fn=collate, drop_last=False)

    amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if device.type == "cuda" and amp_dtype is not torch.float32
                    else torch.autocast(device_type="cpu", enabled=False))

    res = evaluate_split(model, loader, vocab.pad, device, autocast_ctx, args.max_batches)
    res.update(experiment=config.experiment_name, split=split,
               slow_memory=bool(model_args.slow_memory), step=ckpt.get("step"))
    print("[eval] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in res.items()))

    out = Path(args.out) if args.out else config.exp_dir / f"eval-{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
