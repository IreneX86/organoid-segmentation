"""Configuration shared by the three independent 2D segmentation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetSpec:
    """Filesystem locations for one semantic-segmentation target."""

    name: str
    input_relative_dir: Path
    label_relative_dir: Path


TARGET_SPECS: dict[str, TargetSpec] = {
    "mito": TargetSpec(
        name="mito",
        input_relative_dir=Path("label_ref/Mito/mito_input"),
        label_relative_dir=Path("label_ref/Mito/mito_output"),
    ),
    "ld": TargetSpec(
        name="ld",
        input_relative_dir=Path("label_ref/LD/LD_input"),
        label_relative_dir=Path("label_ref/LD/LD_output"),
    ),
    "nuc": TargetSpec(
        name="nuc",
        input_relative_dir=Path("label_ref/Nuc/Nuc_input"),
        label_relative_dir=Path("label_ref/Nuc/Nuc_output"),
    ),
}


@dataclass(frozen=True)
class PipelineConfig:
    """Numerical defaults for patch training and sliding-window inference."""

    patch_size: int = 512
    overlap: int = 128
    lower_percentile: float = 1.0
    upper_percentile: float = 99.8
    threshold: float = 0.5
    random_seed: int = 42

    def validate(self) -> None:
        if self.patch_size <= 0 or self.patch_size % 16 != 0:
            raise ValueError("patch_size must be positive and divisible by 16")
        if not 0 <= self.overlap < self.patch_size:
            raise ValueError("overlap must satisfy 0 <= overlap < patch_size")
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("normalization percentiles are invalid")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between 0 and 1")


def get_target_spec(target: str) -> TargetSpec:
    """Return a target configuration or raise a helpful error."""

    try:
        return TARGET_SPECS[target.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(TARGET_SPECS))
        raise ValueError(f"Unsupported target {target!r}. Choose one of: {supported}") from exc

