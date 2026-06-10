"""Tokenize the multispeaker spoken StoryCloze benchmark into cached units.

Input: parquet shards of `slprl/multispeaker-storycloze` (one speaker subset),
each row = {id, correct_text, correct_audio, negative_text, negative_audio}
where audio is Kokoro-TTS 24 kHz wav bytes. sSC = StoryCloze with the adversarial
wrong ending (fine-grained semantics); tSC = topic-StoryCloze (the wrong ending
comes from another story -- long-range topical coherence, the regime the nested
slow memory targets).

Output: UnitStore splits `<name>_correct` / `<name>_negative` under --out, in
matching row order, encoded with the SAME frozen HuBERT layer + k-means codebook
as training (codebook read from <out>/kmeans.pkl or --codebook).

    python -m speechtext.prepare_storycloze \
        --parquet-dir /ptmp/$USER/LinearAttention/ds/storycloze/sSC/bm \
        --name ssc_bm --out /ptmp/$USER/LinearAttention/units

Needs GPU for HuBERT; ~1871 stories x 2 audios per task.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np

from .corpus import UnitStore
from .tokenizer import DEFAULT_LAYER, HUBERT_SR, HubertFeaturizer, KMeansQuantizer


def decode_audio(blob: dict) -> np.ndarray:
    """parquet audio cell {bytes, path} -> mono 16 kHz float32 waveform."""
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != HUBERT_SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=HUBERT_SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", required=True,
                    help="dir with train-*.parquet of ONE task+speaker (e.g. sSC/bm)")
    ap.add_argument("--name", required=True, help="output split prefix, e.g. ssc_bm")
    ap.add_argument("--out", required=True, help="units dir (kmeans.pkl lives here)")
    ap.add_argument("--codebook", default=None, help="override path to kmeans.pkl")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--limit", type=int, default=None, help="cap #stories (smoke)")
    args = ap.parse_args()

    import pandas as pd

    shards = sorted(Path(args.parquet_dir).glob("*.parquet"))
    assert shards, f"no parquet files under {args.parquet_dir}"
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    if args.limit:
        df = df.iloc[:args.limit]
    print(f"[storycloze] {len(df)} stories from {len(shards)} shards "
          f"({args.parquet_dir})", flush=True)

    feat = HubertFeaturizer(layer=args.layer)
    q = KMeansQuantizer.load(args.codebook or Path(args.out) / "kmeans.pkl")
    store = UnitStore(args.out)

    sides = {"correct": [], "negative": []}
    t0, total = time.time(), 0
    for j, row in enumerate(df.itertuples(index=False)):
        for side in sides:
            wav = decode_audio(getattr(row, f"{side}_audio"))
            units = q.predict(feat.features(wav))
            sides[side].append((str(row.id), str(getattr(row, f"{side}_text")), units))
            total += units.shape[0]
        if (j + 1) % 100 == 0:
            print(f"[storycloze {args.name}] {j+1}/{len(df)} stories, "
                  f"{total} units, {time.time()-t0:.0f}s", flush=True)

    meta = {"layer": args.layer, "n_units": q.n_units, "n_utts": len(df),
            "sr": HUBERT_SR, "source": str(args.parquet_dir)}
    for side, items in sides.items():
        store.write(f"{args.name}_{side}", items, meta={**meta, "side": side})
        print(f"[storycloze] wrote {args.name}_{side}: {len(items)} utts")


if __name__ == "__main__":
    main()
