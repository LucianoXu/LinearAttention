"""Fast, dependency-light correctness tests for the speech-text data layer.

Covers vocab layout, interleaving, and the collate/shift logic -- none of which
need HuBERT, a GPU, or cached units. Run inside the container:

    python -m pytest speechtext/test_speechtext.py -q
"""

import numpy as np
import torch

from .ds import (MOD_PAD, MOD_SPECIAL, MOD_SPEECH, MOD_TEXT, build_sequence,
                 make_collate)
from .vocab import N_BYTES, Vocab


def test_vocab_layout_is_contiguous_and_disjoint():
    v = Vocab(n_units=500)
    assert v.size == N_BYTES + 500 + 4
    # segments don't overlap and cover the range
    assert v.is_byte(0) and v.is_byte(255) and not v.is_byte(256)
    assert v.is_unit(256) and v.is_unit(755) and not v.is_unit(756)
    assert {v.text, v.speech, v.eos, v.pad} == {756, 757, 758, 759}
    # every special is outside byte/unit ranges
    for s in (v.text, v.speech, v.eos, v.pad):
        assert not v.is_byte(s) and not v.is_unit(s)


def test_encode_ranges():
    v = Vocab(n_units=100)
    txt = v.encode_text("Hi!")
    assert txt == [ord("H"), ord("i"), ord("!")]
    assert all(v.is_byte(t) for t in txt)
    units = v.encode_units([0, 99])
    assert units == [256, 256 + 99]
    assert all(v.is_unit(t) for t in units)


def test_build_sequence_both_directions():
    v = Vocab(n_units=100)
    units = np.array([3, 4, 5], dtype=np.uint16)

    toks, mods = build_sequence(v, "ab", units, text_first=True)
    assert toks[0] == v.text and toks[-1] == v.eos
    assert v.speech in toks
    # text tag, 2 bytes, speech tag, 3 units, eos
    assert len(toks) == 1 + 2 + 1 + 3 + 1
    assert mods == [MOD_SPECIAL, MOD_TEXT, MOD_TEXT,
                    MOD_SPECIAL, MOD_SPEECH, MOD_SPEECH, MOD_SPEECH, MOD_SPECIAL]

    toks2, mods2 = build_sequence(v, "ab", units, text_first=False)
    assert toks2[0] == v.speech and toks2[-1] == v.eos
    assert mods2[1] == MOD_SPEECH and mods2[-2] == MOD_TEXT


def test_collate_shifts_and_masks():
    v = Vocab(n_units=100)
    ctx = 16
    collate = make_collate(v, ctx)
    units = np.array([3, 4], dtype=np.uint16)
    toks, mods = build_sequence(v, "ab", units, text_first=True)
    batch = [{"tokens": torch.tensor(toks), "mods": torch.tensor(mods)}]
    out = collate(batch)
    n = len(toks) - 1
    # input/target are a 1-step shift of the same stream
    assert torch.equal(out["input_ids"][0, :n], torch.tensor(toks[:n]))
    assert torch.equal(out["target_ids"][0, :n], torch.tensor(toks[1:n + 1]))
    assert torch.equal(out["target_mod"][0, :n], torch.tensor(mods[1:n + 1]))
    # padding region is pad id / MOD_PAD
    assert (out["target_ids"][0, n:] == v.pad).all()
    assert (out["target_mod"][0, n:] == MOD_PAD).all()


def test_collate_truncates_to_context():
    v = Vocab(n_units=100)
    ctx = 4
    collate = make_collate(v, ctx)
    units = np.arange(50, dtype=np.uint16)
    toks, mods = build_sequence(v, "abcdefgh", units, text_first=True)
    batch = [{"tokens": torch.tensor(toks), "mods": torch.tensor(mods)}]
    out = collate(batch)
    assert out["input_ids"].shape == (1, ctx)
    assert (out["target_mod"][0] != MOD_PAD).all()  # fully filled, no pad
