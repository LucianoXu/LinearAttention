"""Merge several cached UnitStore splits into one combined split.

Used to build the 960h training split from the three LibriSpeech train sets:

    python -m speechtext.merge_units --units-dir /ptmp/$USER/LinearAttention/units \
        --splits train-clean-100 train-clean-360 train-other-500 --name train-960

Utterance order within each split is preserved (sorted by utt_id, i.e.
speaker/chapter-contiguous), so packed windows still see coherent long-range
context; splits are simply concatenated. CPU-only, seconds.
"""

from __future__ import annotations

import argparse

import numpy as np

from .corpus import UnitStore


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units-dir", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--name", required=True, help="output split name")
    args = ap.parse_args()

    store = UnitStore(args.units_dir)
    items, metas = [], []
    for split in args.splits:
        units, index, meta = store.load(split)
        metas.append(meta)
        for rec in index:
            items.append((rec["utt_id"], rec["text"],
                          np.asarray(units[rec["start"]:rec["start"] + rec["length"]])))
        print(f"[merge] {split}: {len(index)} utts, "
              f"{sum(r['length'] for r in index)} units")

    base = {k: metas[0].get(k) for k in ("layer", "n_units", "sr")}
    assert all({k: m.get(k) for k in base} == base for m in metas), \
        f"incompatible split metas: {metas}"
    store.write(args.name, items, meta={**base, "n_utts": len(items),
                                        "n_units_total": int(sum(i[2].shape[0] for i in items)),
                                        "merged_from": args.splits})
    print(f"[merge] wrote {args.name}: {len(items)} utts")


if __name__ == "__main__":
    main()
