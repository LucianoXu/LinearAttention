"""Shared speech-text data layer for the joint spoken-LM project.

Both the flat baseline and the nested model train on the *same* token stream, so
the tokenizer, combined vocabulary, and interleaved dataset live here (one place)
rather than inside each model package.

Pipeline:
    raw audio (.flac)  --HuBERT layer-L features-->  k-means (K units)  -->  unit ids
    raw text           --bytes-->                    byte ids (0..255)
    interleave with modality tags  -->  single autoregressive token stream

See `vocab.py` for the combined token layout, `tokenizer.py` for the speech
tokenizer, `prepare_units.py` for the one-shot caching script, and `ds.py` for
the training dataset.
"""

from .vocab import Vocab

__all__ = ["Vocab"]
