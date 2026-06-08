"""Interleaved speech-text dataset for the joint spoken LM.

Each training example is a single autoregressive token stream that places one
utterance's text span and speech-unit span back to back, in a random order:

    <text>  b b b ... <speech>  u u u ... <eos>          (text -> speech)
    <speech> u u u ... <text>  b b b ... <eos>           (speech -> text)

Random direction per example forces the model to predict speech from text *and*
text from speech, so the cross-modal conditional loss is meaningful in both
directions. No forced alignment is needed.

Alongside the token ids we carry a per-position **modality code** so evaluation
can split next-token loss by what is being predicted:

    MOD_TEXT (0)   target is a byte token
    MOD_SPEECH (1) target is a speech-unit token
    MOD_SPECIAL(2) target is a tag / <eos>
    MOD_PAD (-1)   padding (ignored by the loss)

The collate fn pads to a fixed `context_len` and builds shifted (input, target)
pairs; `target` uses `pad` as the ignore id and a parallel `target_mod` array
carries the modality of each target for masked per-modality metrics.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .corpus import UnitStore
from .vocab import Vocab

MOD_TEXT, MOD_SPEECH, MOD_SPECIAL, MOD_PAD = 0, 1, 2, -1


def build_sequence(vocab: Vocab, text: str, units: np.ndarray,
                   text_first: bool) -> tuple[list[int], list[int]]:
    """Return (token_ids, modality_codes) for one interleaved utterance."""
    text_tok = vocab.encode_text(text)
    unit_tok = vocab.encode_units(units)

    text_block = ([vocab.text] + text_tok, [MOD_SPECIAL] + [MOD_TEXT] * len(text_tok))
    speech_block = ([vocab.speech] + unit_tok, [MOD_SPECIAL] + [MOD_SPEECH] * len(unit_tok))

    first, second = (text_block, speech_block) if text_first else (speech_block, text_block)
    toks = first[0] + second[0] + [vocab.eos]
    mods = first[1] + second[1] + [MOD_SPECIAL]
    return toks, mods


class SpokenLMDataset(Dataset):
    """One interleaved utterance per item, truncated to `context_len`.

    Packing multiple utterances per window is a later optimization; one
    utterance per item keeps the speech/text spans clean for per-modality loss
    and is plenty for the ablation.
    """

    def __init__(self, units_dir: str, split: str, vocab: Vocab,
                 context_len: int = 2048, seed: int = 42):
        self.vocab = vocab
        self.context_len = context_len
        units, index, meta = UnitStore(units_dir).load(split)
        self._units = units
        self._index = index
        self.meta = meta
        assert meta["n_units"] == vocab.n_units, \
            f"vocab n_units={vocab.n_units} != cached {meta['n_units']}"
        self._base_seed = seed

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        rec = self._index[i]
        units = np.asarray(self._units[rec["start"]:rec["start"] + rec["length"]])
        # deterministic-but-varied direction per (item, epoch-ish) without global state
        rng = np.random.default_rng(self._base_seed + i)
        text_first = bool(rng.integers(2))
        toks, mods = build_sequence(self.vocab, rec["text"], units, text_first)
        toks = toks[: self.context_len + 1]      # +1 so shift still yields context_len
        mods = mods[: self.context_len + 1]
        return {"tokens": torch.tensor(toks, dtype=torch.long),
                "mods": torch.tensor(mods, dtype=torch.long)}


def make_collate(vocab: Vocab, context_len: int):
    """Pad to `context_len` and produce shifted (input, target, target_mod)."""
    pad = vocab.pad

    def collate(batch):
        B = len(batch)
        x = torch.full((B, context_len), pad, dtype=torch.long)
        y = torch.full((B, context_len), pad, dtype=torch.long)
        ymod = torch.full((B, context_len), MOD_PAD, dtype=torch.long)
        for b, item in enumerate(batch):
            t, m = item["tokens"], item["mods"]
            n = min(t.shape[0] - 1, context_len)     # predict positions 1..n
            x[b, :n] = t[:n]
            y[b, :n] = t[1:n + 1]
            ymod[b, :n] = m[1:n + 1]
        return {"input_ids": x, "target_ids": y, "target_mod": ymod}

    return collate
