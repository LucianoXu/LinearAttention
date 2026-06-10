"""Spoken StoryCloze (sSC / tSC) likelihood evaluation.

Standard SpeechLM benchmark protocol (GSLM / TWIST / SLAM): each story exists
in two spoken versions -- with the correct ending and with a wrong ending. The
model scores both unit sequences; accuracy = fraction of stories where the
correct version gets the higher log-likelihood. Chance = 50%.

  - sSC (spoken StoryCloze): the wrong ending is the adversarial human-written
    one -- needs fine-grained semantics.
  - tSC (topic StoryCloze): the wrong ending is sampled from another story --
    needs topical / long-range coherence, exactly what a slow memory should buy.

Sequences are rendered the way training rendered speech spans:
`<speech> u u u ... <eos>`; log-likelihood is summed over the unit tokens and
<eos> (the prediction targets), reported both summed and per-token.

    python -m nestedla.eval_storycloze nestedla/config_nested_packed360.yaml \
        --ckpt .../ckpt-12000.pth --units-dir .../units --task ssc_bm tsc_bm
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from speechtext.corpus import UnitStore
from speechtext.vocab import Vocab
from .config import TrainConfig
from .model import NestedLA, ModelArgs


def render(vocab: Vocab, units: np.ndarray, max_len: int) -> list[int]:
    """<speech> u u u ... <eos>, truncated (keeping <eos>) to max_len tokens."""
    toks = [vocab.speech] + vocab.encode_units(units) + [vocab.eos]
    if len(toks) > max_len:
        toks = toks[: max_len - 1] + [vocab.eos]
    return toks


@torch.no_grad()
def sequence_logprobs(model, seqs, vocab, device, autocast_ctx, chunk: int,
                      batch_size: int = 16):
    """Per-sequence (sum_logprob, n_targets) under the causal LM.

    Right-pads each batch to a multiple of `chunk` (the chunkwise kernel needs
    L % chunk == 0; right-padding is causal-safe).
    """
    order = np.argsort([len(s) for s in seqs])  # batch similar lengths
    sums = np.zeros(len(seqs)); ns = np.zeros(len(seqs), dtype=np.int64)
    for b0 in range(0, len(order), batch_size):
        ids = order[b0:b0 + batch_size]
        batch = [seqs[i] for i in ids]
        L = max(len(s) for s in batch) - 1
        L = ((L + chunk - 1) // chunk) * chunk
        x = torch.full((len(batch), L), vocab.pad, dtype=torch.long)
        y = torch.full((len(batch), L), vocab.pad, dtype=torch.long)
        for r, s in enumerate(batch):
            t = torch.tensor(s, dtype=torch.long)
            x[r, : len(s) - 1] = t[:-1]
            y[r, : len(s) - 1] = t[1:]
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            logits = model(x)
            B, T, V = logits.shape
            ce = nn.functional.cross_entropy(
                logits.reshape(B * T, V), y.reshape(B * T),
                ignore_index=vocab.pad, reduction="none").reshape(B, T)
        valid = (y != vocab.pad)
        lp = -(ce * valid).sum(dim=1)
        for r, i in enumerate(ids):
            sums[i] = lp[r].item()
            ns[i] = int(valid[r].sum().item())
    return sums, ns


def main():
    ap = argparse.ArgumentParser(description="Spoken StoryCloze eval")
    ap.add_argument("config")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--units-dir", default=None, help="default: config units_dir")
    ap.add_argument("--task", nargs="+", default=["ssc_bm", "tsc_bm"],
                    help="UnitStore split prefixes (expects <task>_correct/_negative)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=None,
                    help="cap sequence tokens (default: model context_len)")
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
    print(f"[storycloze] loaded {args.ckpt} (step {ckpt.get('step')}), "
          f"slow_memory={model_args.slow_memory}")

    vocab = Vocab(n_units=model_args.vocab_size - 256 - 4)
    store = UnitStore(args.units_dir or config.units_dir)
    max_len = args.max_len or model_args.context_len
    amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if device.type == "cuda" and amp_dtype is not torch.float32
                    else torch.autocast(device_type="cpu", enabled=False))

    res = {"experiment": config.experiment_name,
           "slow_memory": bool(model_args.slow_memory), "step": ckpt.get("step")}
    for task in args.task:
        sides = {}
        for side in ("correct", "negative"):
            units, index, _ = store.load(f"{task}_{side}")
            seqs = [render(vocab,
                           np.asarray(units[r["start"]:r["start"] + r["length"]]),
                           max_len) for r in index]
            sides[side] = sequence_logprobs(model, seqs, vocab, device,
                                            autocast_ctx, model_args.chunk_size,
                                            args.batch_size)
        (lc, nc), (ln, nn_) = sides["correct"], sides["negative"]
        acc_sum = float(np.mean(lc > ln))
        acc_mean = float(np.mean(lc / np.maximum(nc, 1) > ln / np.maximum(nn_, 1)))
        res[task] = {"n": len(lc), "acc_sum": round(acc_sum, 4),
                     "acc_mean": round(acc_mean, 4),
                     "mean_lp_correct": round(float(np.mean(lc / nc)), 4),
                     "mean_lp_negative": round(float(np.mean(ln / nn_)), 4)}
        print(f"[storycloze] {task}: n={len(lc)} acc(sum)={acc_sum:.4f} "
              f"acc(per-token)={acc_mean:.4f}")

    out = Path(args.out) if args.out else config.exp_dir / "eval-storycloze.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[storycloze] wrote {out}")


if __name__ == "__main__":
    main()
