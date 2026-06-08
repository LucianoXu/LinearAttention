"""One-shot: turn LibriSpeech audio into cached k-means unit streams.

Two stages:
  1. fit   -- extract HuBERT features on a subset (~1-2 h of audio) and fit a
              MiniBatchKMeans codebook (saved to <out>/kmeans.pkl).
  2. encode -- run HuBERT + the codebook over every utterance of one or more
              splits, caching units via UnitStore.

Runs INSIDE the container (HuBERT needs torch); GPU strongly recommended for the
encode pass. HuBERT weights are read from the local HF cache (set HF_HOME), so
this never touches the network -- safe on a compute node.

Examples (from project root, inside the container):
    python -m speechtext.prepare_units fit \
        --root /ptmp/$USER/LinearAttention/ds/LibriSpeech --fit-split dev-clean \
        --fit-hours 1.5 --out /ptmp/$USER/LinearAttention/units
    python -m speechtext.prepare_units encode \
        --root /ptmp/$USER/LinearAttention/ds/LibriSpeech \
        --splits dev-clean train-clean-100 \
        --out /ptmp/$USER/LinearAttention/units
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .corpus import UnitStore, scan_split
from .tokenizer import DEFAULT_LAYER, HUBERT_SR, HubertFeaturizer, KMeansQuantizer, load_audio


def _fit(args):
    feat = HubertFeaturizer(layer=args.layer)
    utts = scan_split(args.root, args.fit_split)

    # take utterances until we reach the requested #hours of audio
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(utts))
    budget_frames = int(args.fit_hours * 3600 * HUBERT_SR / 320)  # ~50 Hz
    feats, got = [], 0
    for i in order:
        f = feat.features(load_audio(utts[i].flac))
        feats.append(f)
        got += f.shape[0]
        if got >= budget_frames:
            break
    X = np.concatenate(feats).astype(np.float32)
    print(f"[fit] {X.shape[0]} frames "
          f"(~{X.shape[0]*320/HUBERT_SR/3600:.2f} h), dim={X.shape[1]}")

    t0 = time.time()
    q = KMeansQuantizer(n_units=args.n_units, seed=args.seed).fit(X)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    q.save(Path(args.out) / "kmeans.pkl")
    print(f"[fit] k-means K={args.n_units} fitted in {time.time()-t0:.1f}s "
          f"-> {Path(args.out) / 'kmeans.pkl'}")


def _encode(args):
    feat = HubertFeaturizer(layer=args.layer)
    q = KMeansQuantizer.load(Path(args.out) / "kmeans.pkl")
    store = UnitStore(args.out)

    for split in args.splits:
        utts = scan_split(args.root, split)
        items, total = [], 0
        t0 = time.time()
        for j, u in enumerate(utts):
            units = q.predict(feat.features(load_audio(u.flac)))
            items.append((u.utt_id, u.text, units))
            total += units.shape[0]
            if (j + 1) % 200 == 0:
                print(f"[encode {split}] {j+1}/{len(utts)} utts, "
                      f"{total} units, {time.time()-t0:.0f}s", flush=True)
        store.write(split, items, meta={
            "layer": args.layer, "n_units": args.n_units,
            "n_utts": len(utts), "n_units_total": total, "sr": HUBERT_SR,
        })
        print(f"[encode {split}] DONE {len(utts)} utts, {total} units "
              f"-> {split}.units.npy ({time.time()-t0:.0f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", required=True, help="LibriSpeech root dir")
    common.add_argument("--out", required=True, help="output dir for kmeans + units")
    common.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    common.add_argument("--n-units", type=int, default=500)
    common.add_argument("--seed", type=int, default=42)

    pf = sub.add_parser("fit", parents=[common])
    pf.add_argument("--fit-split", default="dev-clean")
    pf.add_argument("--fit-hours", type=float, default=1.5)
    pf.set_defaults(func=_fit)

    pe = sub.add_parser("encode", parents=[common])
    pe.add_argument("--splits", nargs="+", required=True)
    pe.set_defaults(func=_encode)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
