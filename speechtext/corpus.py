"""LibriSpeech manifest scanning and on-disk unit storage.

LibriSpeech layout:
    <root>/<split>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
    <root>/<split>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt
      ... each .trans.txt line: "<speaker>-<chapter>-<utt> THE TRANSCRIPT TEXT"

Units are cached as a single concatenated uint16 array plus a JSON index, so a
split loads via one memmap (fast, few inodes) instead of tens of thousands of
tiny files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Utterance:
    utt_id: str
    flac: Path
    text: str


def scan_split(root: str | Path, split: str) -> list[Utterance]:
    """Return all utterances of a LibriSpeech split, sorted by utt_id."""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split dir not found: {split_dir}")

    # gather transcripts: utt_id -> text
    texts: dict[str, str] = {}
    for trans in split_dir.rglob("*.trans.txt"):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, _, text = line.partition(" ")
            texts[uid] = text

    utts: list[Utterance] = []
    for flac in split_dir.rglob("*.flac"):
        uid = flac.stem
        if uid in texts:
            utts.append(Utterance(uid, flac, texts[uid]))
    utts.sort(key=lambda u: u.utt_id)
    return utts


class UnitStore:
    """Concatenated uint16 units + index.

    Files written under `out_dir`:
        <split>.units.npy   concatenated unit ids (uint16)
        <split>.index.json  [{utt_id, text, start, length}, ...] + meta
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)

    def _paths(self, split: str) -> tuple[Path, Path]:
        return (self.out_dir / f"{split}.units.npy",
                self.out_dir / f"{split}.index.json")

    def write(self, split: str, items: list[tuple[str, str, np.ndarray]],
              meta: dict) -> None:
        """items: list of (utt_id, text, unit_ids[uint16])."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        units_path, index_path = self._paths(split)

        index = []
        start = 0
        chunks = []
        for utt_id, text, units in items:
            units = np.asarray(units, dtype=np.uint16)
            chunks.append(units)
            index.append({"utt_id": utt_id, "text": text,
                          "start": start, "length": int(units.shape[0])})
            start += int(units.shape[0])

        all_units = (np.concatenate(chunks) if chunks
                     else np.zeros(0, dtype=np.uint16))
        np.save(units_path, all_units)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "index": index}, f)

    def load(self, split: str) -> tuple[np.ndarray, list[dict], dict]:
        """Return (units_memmap, index, meta)."""
        units_path, index_path = self._paths(split)
        units = np.load(units_path, mmap_mode="r")
        with open(index_path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return units, blob["index"], blob["meta"]
