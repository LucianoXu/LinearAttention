"""Frozen speech tokenizer: HuBERT features -> k-means discrete units.

GSLM/SpeechGPT-style. We take a single hidden layer of a frozen HuBERT-Base
model (~50 Hz frame rate for 16 kHz audio) and quantise each frame to its
nearest k-means centroid. The result is a single stream of `n_units` discrete
ids per utterance -- dense relative to text, which is exactly the density
contrast the nested-memory experiment relies on (no de-duplication of repeated
units).

HuBERT weights are loaded from the local HF cache (HF_HOME); compute nodes have
no internet, so the weights must be pre-fetched on a login node (see RAVEN
staging step). Audio is read with soundfile (LibriSpeech is already 16 kHz mono
.flac, so no resampling is needed in the common path).

This module deliberately avoids torchaudio (absent from the MPCDF container):
soundfile + transformers + sklearn are all present / pip-installable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

HUBERT_NAME = "facebook/hubert-base-ls960"
HUBERT_SR = 16_000          # HuBERT expects 16 kHz mono
DEFAULT_LAYER = 9           # transformer layer to read features from (1..12)


def load_audio(path: str | Path) -> np.ndarray:
    """Read a mono 16 kHz waveform as float32 in [-1, 1].

    LibriSpeech flac is already 16 kHz mono; we only resample / downmix on the
    rare path where it is not, and import librosa lazily for that.
    """
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:                       # stereo -> mono
        wav = wav.mean(axis=1)
    if sr != HUBERT_SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=HUBERT_SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


class HubertFeaturizer:
    """Frozen HuBERT feature extractor (one chosen hidden layer)."""

    def __init__(self, layer: int = DEFAULT_LAYER, device: str | None = None,
                 dtype: torch.dtype = torch.float32):
        from transformers import HubertModel

        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.model = HubertModel.from_pretrained(HUBERT_NAME)
        self.model.eval().to(self.device, dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def features(self, wav: np.ndarray) -> np.ndarray:
        """waveform (T,) float32 -> features (n_frames, 768) float32.

        n_frames ~= len(wav) / 320  (i.e. ~50 Hz).
        """
        x = torch.from_numpy(wav).to(self.device, dtype=self.dtype).unsqueeze(0)
        out = self.model(x, output_hidden_states=True)
        # hidden_states: tuple of (embeddings, layer1, ..., layer12); index `layer`
        # selects transformer layer `layer` (1-based) directly.
        feat = out.hidden_states[self.layer].squeeze(0)
        return feat.float().cpu().numpy()


class KMeansQuantizer:
    """Thin wrapper over sklearn MiniBatchKMeans for fit / predict / save / load.

    MiniBatchKMeans scales to the ~millions of frames produced by an hour-plus of
    audio without holding everything in one giant matmul.
    """

    def __init__(self, n_units: int = 500, seed: int = 42):
        self.n_units = n_units
        self.seed = seed
        self.km = None                     # set by fit() / load()

    def fit(self, features: np.ndarray, batch_size: int = 10_000,
            max_iter: int = 100) -> "KMeansQuantizer":
        from sklearn.cluster import MiniBatchKMeans

        self.km = MiniBatchKMeans(
            n_clusters=self.n_units,
            random_state=self.seed,
            batch_size=batch_size,
            max_iter=max_iter,
            n_init=5,
            max_no_improvement=50,
            verbose=0,
        )
        self.km.fit(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.km is not None, "quantizer not fitted/loaded"
        return self.km.predict(features).astype(np.uint16)

    def save(self, path: str | Path):
        import pickle

        assert self.km is not None
        with open(path, "wb") as f:
            pickle.dump({"n_units": self.n_units, "seed": self.seed,
                         "centroids": self.km.cluster_centers_}, f)

    @classmethod
    def load(cls, path: str | Path) -> "KMeansQuantizer":
        import pickle

        from sklearn.cluster import MiniBatchKMeans

        with open(path, "rb") as f:
            d = pickle.load(f)
        q = cls(n_units=d["n_units"], seed=d["seed"])
        # rebuild a predict-only MiniBatchKMeans from saved centroids
        km = MiniBatchKMeans(n_clusters=d["n_units"], random_state=d["seed"])
        km.cluster_centers_ = d["centroids"]
        km._n_threads = 1
        q.km = km
        return q
