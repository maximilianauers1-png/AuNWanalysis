"""Central configuration objects.

Every tunable number of the pipeline lives here, grouped by stage. Dataclasses
make the settings explicit, type-checked by your editor, and trivially
serialisable to JSON (``dataclasses.asdict``), so every trained model and
every analysis export carries the exact parameters that produced it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Type, TypeVar

T = TypeVar("T")

# Project folders (relative to the repository root).
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


@dataclass
class SyntheticConfig:
    """Parameters of the synthetic SEM nanowire-mesh generator.

    All lengths are in *pixels of the network input*. Real images are rescaled
    at inference time so that wires fall into the same width range.
    """

    tile_size: int = 256
    n_wires: tuple[int, int] = (6, 45)            # wires per tile (uniform)
    length_median: tuple[float, float] = (40.0, 320.0)  # median of log-normal length
    length_sigma: tuple[float, float] = (0.25, 0.6)     # log-normal shape parameter
    width_mean: tuple[float, float] = (3.0, 20.0)       # per-image mean width
    width_cv: tuple[float, float] = (0.03, 0.20)        # per-image coefficient of variation
    max_bend_deg: float = 35.0       # total turning angle of a wire (circular arc)
    edge_gain: tuple[float, float] = (0.15, 0.6)  # strength of the SE edge effect
    particle_prob: float = 0.5       # probability that a tile contains Au particles
    pore_prob: float = 0.25          # probability of a porous (filter-membrane) substrate
    blur_sigma: tuple[float, float] = (0.4, 1.4)  # electron-probe PSF (Gaussian)
    photons: tuple[float, float] = (15.0, 250.0)  # Poisson dose; lower = noisier
    jpeg_prob: float = 0.3           # probability of JPEG re-compression artefacts
    # dense 3-D meshes (in-lens detector look): depth layers, defocus, faceted ribbons
    dense_prob: float = 0.65                         # fraction of images in this style
    dense_layers: tuple[int, int] = (3, 7)           # number of depth layers
    dense_wires_per_layer: tuple[int, int] = (3, 9)  # per 256^2 tile
    dense_length_median: tuple[float, float] = (150.0, 900.0)  # long: most cross the tile
    dense_width_mean: tuple[float, float] = (6.0, 22.0)


@dataclass
class ModelConfig:
    """Architecture of the residual U-Net."""

    base_filters: int = 24     # filters of the first encoder level (doubles per level)
    depth: int = 4             # number of down-sampling steps
    dropout: float = 0.15      # SpatialDropout2D rate in the deep levels
    tile_size: int = 256       # training tile size (inference works on any size)

    @property
    def filters(self) -> list[int]:
        return [self.base_filters * 2**i for i in range(self.depth + 1)]


@dataclass
class TrainConfig:
    """Optimisation and regularisation settings."""

    epochs: int = 60
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4        # decoupled weight decay (AdamW)
    early_stop_patience: int = 10     # epochs without val improvement before stopping
    lr_patience: int = 4              # epochs before the learning rate is halved
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"mask": 1.0, "centerline": 1.0, "orientation": 0.5, "width": 0.5}
    )
    augment: bool = True
    mixed_precision: bool = False     # only worthwhile on a recent GPU
    seed: int = 42


@dataclass
class AnalysisConfig:
    """Post-processing that turns network maps into individual wire measurements."""

    mask_threshold: float = 0.5
    centerline_threshold: float = 0.5
    tile_overlap: int = 64            # overlap of sliding-window tiles (px)
    tta: bool = True                  # test-time augmentation (4 flips)
    auto_scale: bool = True           # rescale image so wires are ~target_width px wide
    target_width_px: float = 12.0
    junction_radius_px: float | None = None  # radius of junction zones (None: from the wire width)
    max_pair_angle_deg: float = 25.0  # max direction change when following a wire through a junction
    min_length_px: float = 12.0       # discard shorter objects (network pixels)
    min_aspect_ratio: float = 3.0     # length/width below this is treated as a particle
    exclude_border_from_length: bool = True  # wires cut by the image border have unknown length
    coverage_reliable: float = 0.30   # above this areal coverage many wire ends are hidden ...
    coverage_limit: float = 0.45      # ... and above this length statistics are unreliable


# --------------------------------------------------------------------------- I/O
def to_dict(cfg: Any) -> dict:
    return asdict(cfg)


def from_dict(cls: Type[T], data: dict) -> T:
    """Build a dataclass from a dict, ignoring unknown keys (forward compatible)."""
    names = {f.name for f in fields(cls)}
    kwargs = {k: tuple(v) if isinstance(v, list) and k not in ("loss_weights",) else v
              for k, v in data.items() if k in names}
    return cls(**kwargs)


def save_json(path: Path, **configs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v)
               for k, v in configs.items()}
    path.write_text(json.dumps(payload, indent=2))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
