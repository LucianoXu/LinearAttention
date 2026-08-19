"""Combined byte + speech-unit + special-token vocabulary.

A single embedding table and a single softmax cover both modalities. The layout
is contiguous so an id alone tells you its modality (used for per-modality loss
masking in evaluation):

    [ 0 .. 255 ]                      byte tokens          (text)
    [ 256 .. 256+n_units-1 ]          speech unit tokens   (speech)
    [ ... 4 trailing ids ... ]        <text> <speech> <eos> <pad>  (specials)

`<text>` / `<speech>` are modality tags emitted right before a span of that
modality; `<eos>` ends a sequence; `<pad>` fills a batch (and is ignored by the
loss). Keeping specials *last* means adding more units never renumbers them.
"""

from dataclasses import dataclass

N_BYTES = 256
SPECIAL_NAMES = ("text", "speech", "eos", "pad")


@dataclass(frozen=True)
class Vocab:
    n_units: int = 500

    # ---- segment boundaries ----
    @property
    def byte_base(self) -> int:
        return 0

    @property
    def unit_base(self) -> int:
        return N_BYTES

    @property
    def special_base(self) -> int:
        return N_BYTES + self.n_units

    @property
    def size(self) -> int:
        return N_BYTES + self.n_units + len(SPECIAL_NAMES)

    # ---- special token ids ----
    @property
    def text(self) -> int:
        return self.special_base + 0

    @property
    def speech(self) -> int:
        return self.special_base + 1

    @property
    def eos(self) -> int:
        return self.special_base + 2

    @property
    def pad(self) -> int:
        return self.special_base + 3

    # ---- id <-> token helpers ----
    def byte_id(self, b: int) -> int:
        assert 0 <= b < N_BYTES
        return self.byte_base + b

    def unit_id(self, u: int) -> int:
        assert 0 <= u < self.n_units
        return self.unit_base + u

    def is_byte(self, tok: int) -> bool:
        return self.byte_base <= tok < self.unit_base

    def is_unit(self, tok: int) -> bool:
        return self.unit_base <= tok < self.special_base

    def encode_text(self, text: str) -> list[int]:
        """UTF-8 bytes -> byte token ids."""
        return [self.byte_base + b for b in text.encode("utf-8")]

    def encode_units(self, units) -> list[int]:
        """k-means unit ids (0..n_units-1) -> unit token ids."""
        return [self.unit_base + int(u) for u in units]
