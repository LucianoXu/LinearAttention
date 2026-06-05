from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .model import ModelArgs


@dataclass
class TrainConfig:
    """Typed training configuration, loaded from a YAML file.

    Defaults live here in one place; a YAML file only needs to override what
    differs from them. Access fields as `config.lr` (typed, IDE-completable)
    instead of `config['lr']`.
    """

    model_args: ModelArgs = field(default_factory=ModelArgs)

    lr: float = 1e-4
    wd: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    grad_norm_clip: float = 1.0
    warm_up_steps: int = 300
    step_limit: int = 2000

    batch_size: int = 4
    seed: int = 42
    train_ratio: float = 0.9

    device: str = "cpu"
    output_path: str = "ckpt/"
    experiment_name: str = "example"

    save_interval: int | None = None
    valid_interval: int = 50
    single_batch_test: bool = False

    def __post_init__(self):
        # YAML gives model_args as a nested dict; build the real dataclass.
        if isinstance(self.model_args, dict):
            self.model_args = ModelArgs(**self.model_args)

        # YAML quirk: `1e-4` (no dot, no sign on the exponent) parses as a
        # *string*, and `betas` arrives as a list. Coerce numeric fields to
        # their real types so the rest of the code can trust them.
        self.lr = float(self.lr)
        self.grad_norm_clip = float(self.grad_norm_clip)
        self.train_ratio = float(self.train_ratio)
        b1, b2 = self.betas
        self.betas = (float(b1), float(b2))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # An unknown key here raises TypeError -> catches typos in the YAML
        # instead of silently ignoring them.
        return cls(**raw)

    @property
    def exp_dir(self) -> Path:
        return Path(self.output_path) / self.experiment_name

    def to_plain_dict(self) -> dict:
        d = asdict(self)                 # recurses into ModelArgs
        d["betas"] = list(d["betas"])    # yaml.safe_dump can't represent tuples
        return d

    def save(self):
        """Dump the *resolved* config (defaults merged in) into the experiment
        directory, so any run can be reproduced from exactly what it used."""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        with open(self.exp_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_plain_dict(), f, sort_keys=False, allow_unicode=True)
