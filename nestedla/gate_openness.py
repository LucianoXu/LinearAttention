"""Measure the learned slow-read gate openness of a nested checkpoint.

Loads a nested (slow_memory=True) checkpoint, runs a few dev batches, and reports
the mean (and per-head) opening of the per-head slow-read gate
`sigmoid(slow_gate(x))`, averaged over all valid (non-pad) token positions.
init = sigmoid(slow_gate_bias_init); a model that "declines to use" the slow
level stays near that init.

    python -m nestedla.gate_openness nestedla/config_nested_packed360fix.yaml \
        --ckpt /ptmp/$USER/LinearAttention/ckpt/nestedla-nested-packed360fix/ckpt-12000.pth
"""

import argparse
import math

import torch
from torch.utils.data import DataLoader

from speechtext.ds import SpokenLMDataset, make_collate
from speechtext.vocab import Vocab
from .config import TrainConfig
from .model import NestedLA, ModelArgs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    config = TrainConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    margs = ckpt["args"] if isinstance(ckpt.get("args"), ModelArgs) else config.model_args
    assert margs.slow_memory, "gate openness only defined for slow_memory=True models"
    model = NestedLA(margs).to(device).eval()
    model.load_state_dict(ckpt["model"])
    init = 1.0 / (1.0 + math.exp(-margs.slow_gate_bias_init))
    print(f"[gate] {args.ckpt} step={ckpt.get('step')} init_open={init:.4f} "
          f"use_rope={getattr(margs, 'use_rope', None)}")

    H = margs.head
    gate_sum = torch.zeros(H, device=device)
    gate_cnt = torch.zeros(H, device=device)
    # hook every layer's slow_gate; accumulate sigmoid over valid token positions
    gates = []

    def mk_hook():
        def hook(module, inp, out):
            gates.append(torch.sigmoid(out.detach().float()))  # (B,L,H)
        return hook

    handles = [blk.att.slow_gate.register_forward_hook(mk_hook()) for blk in model.blocks]

    vocab = Vocab(n_units=margs.vocab_size - 256 - 4)
    collate = make_collate(vocab, margs.context_len)
    ds = SpokenLMDataset(config.units_dir, args.split, vocab,
                         context_len=margs.context_len, seed=config.seed + 7,
                         pack=getattr(config, "pack", False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate, drop_last=False)

    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        x = batch["input_ids"].to(device)
        ymod = batch["target_mod"].to(device)
        valid = (ymod != -1)                                  # (B,L)
        gates.clear()
        with amp:
            model(x)
        for g in gates:                                       # one per layer
            gv = g[valid]                                     # (Nvalid, H)
            gate_sum += gv.sum(0)
            gate_cnt += gv.shape[0]
    for h in handles:
        h.remove()

    per_head = (gate_sum / gate_cnt.clamp(min=1)).tolist()
    overall = float(gate_sum.sum() / gate_cnt.sum().clamp(min=1))
    print(f"[gate] mean_open={overall:.4f} (init {init:.4f})  per_head=" +
          " ".join(f"{v:.3f}" for v in per_head))


if __name__ == "__main__":
    main()
