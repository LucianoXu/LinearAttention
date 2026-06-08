from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .model import ModelArgs


@dataclass
class TrainConfig:
    """Typed training configuration for the nested speech-text LM.

    Mirrors transformer/config.py but points at the cached speech-unit store
    (speechtext) instead of a byte-corpus .txt.
    """

    model_args: ModelArgs = field(default_factory=ModelArgs)

    lr: float = 3e-4
    wd: float = 0.05
    betas: tuple[float, float] = (0.9, 0.98)
    grad_norm_clip: float = 1.0
    warm_up_steps: int = 300
    step_limit: int = 5000

    batch_size: int = 16           # PER-GPU micro-batch size
    grad_accum_steps: int = 1
    eval_batch_size: int = 16
    seed: int = 42

    # ---- data (speechtext unit store) ----
    units_dir: str = "/ptmp/yinxu/LinearAttention/units"
    train_split: str = "train-clean-100"
    valid_split: str = "dev-clean"

    device: str = "cpu"            # ignored under DDP/torchrun
    output_path: str = "ckpt/"
    experiment_name: str = "nestedla"

    # ---- efficiency knobs (A100 / multi-GPU) ----
    dtype: str = "bfloat16"
    compile: bool = True
    tf32: bool = True
    num_workers: int = 8

    save_interval: int | None = 2000
    valid_interval: int = 200
    valid_batches: int = 20
    log_interval: int = 10
    single_batch_test: bool = False

    def __post_init__(self):
        if isinstance(self.model_args, dict):
            self.model_args = ModelArgs(**self.model_args)
        self.lr = float(self.lr)
        self.grad_norm_clip = float(self.grad_norm_clip)
        b1, b2 = self.betas
        self.betas = (float(b1), float(b2))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    @property
    def exp_dir(self) -> Path:
        return Path(self.output_path) / self.experiment_name

    def to_plain_dict(self) -> dict:
        d = asdict(self)
        d["betas"] = list(d["betas"])
        return d

    def save(self):
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        with open(self.exp_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_plain_dict(), f, sort_keys=False, allow_unicode=True)
