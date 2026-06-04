from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class TrainingConfig:
    """Schema cho DistilBERT Extractive QA config.

    Gia tri config nam trong YAML. Class nay chi validate key va cung cap
    object typed de trainer su dung.
    """

    model_name: str
    init_checkpoint_dir: Optional[str]
    dropout: float
    freeze_encoder: bool

    dataset_name: Optional[str]
    dataset_config_name: Optional[str]
    train_file: Optional[str]
    validation_file: Optional[str]
    test_file: Optional[str]

    question_column: str
    context_column: str
    answers_column: str
    impossible_column: str
    plausible_answers_column: str

    max_length: int
    doc_stride: int
    padding: str
    cache_dir: Optional[str]

    use_vietnamese_segmentation: bool
    segmentation_tool: str

    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    gradient_accumulation_steps: int

    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    use_amp: bool
    use_tf32: bool
    force_cpu: bool

    eval_steps: Optional[int]
    save_steps: Optional[int]
    eval_strategy: str
    save_strategy: str
    save_best_model: bool
    best_metric: str
    load_best_model: bool

    output_dir: str
    artifact_dir: str
    artifact_name: str
    onnx_opset_version: int
    seed: int
    logging_steps: int
    log_level: str

    @classmethod
    def from_yaml(cls, path: str | Path = "config/model.yaml", profile: str | None = None) -> "TrainingConfig":
        """Load config tu YAML.

        YAML co the la mapping phang, hoac gom `defaults` + `profiles`.
        Khi co profiles, truyen `profile="train_vi"`, `"train_en"`,
        `"train_mixed"`, `"eval_vi"`... Neu khong truyen profile thi dung
        `default_profile` trong YAML.
        """
        data = _load_yaml(Path(path))
        config = _resolve_profile_config(data, profile)
        return cls(**_validate_config(config, cls._field_names(), Path(path)))

    @classmethod
    def available_profiles(cls, path: str | Path = "config/model.yaml") -> list[str]:
        """Tra ve danh sach profile trong YAML config."""
        data = _load_yaml(Path(path))
        profiles = data.get("profiles", {})
        if not isinstance(profiles, dict):
            raise ValueError("`profiles` must be a YAML mapping.")
        return sorted(profiles)

    def to_yaml(self, path: str | Path) -> None:
        """Luu config da resolve ra file YAML phang."""
        Path(path).write_text(
            yaml.dump(data=asdict(self), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def _field_names(cls) -> set[str]:
        return {field.name for field in fields(cls)}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Training config must be a YAML mapping: {path}")
    return data


def _resolve_profile_config(data: dict[str, Any], profile: str | None) -> dict[str, Any]:
    if "profiles" not in data:
        return data

    defaults = data.get("defaults", {})
    profiles = data.get("profiles", {})
    if not isinstance(defaults, dict):
        raise ValueError("`defaults` must be a YAML mapping.")
    if not isinstance(profiles, dict):
        raise ValueError("`profiles` must be a YAML mapping.")

    selected_profile = profile or data.get("default_profile")
    if not selected_profile:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Config profile is required. Available profiles: {available}")
    if selected_profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown config profile `{selected_profile}`. Available profiles: {available}")
    if not isinstance(profiles[selected_profile], dict):
        raise ValueError(f"Profile `{selected_profile}` must be a YAML mapping.")

    return defaults | profiles[selected_profile]


def _validate_config(config: dict[str, Any], valid_keys: set[str], path: Path) -> dict[str, Any]:
    unknown_keys = sorted(set(config) - valid_keys)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unknown training config key(s) in {path}: {joined}")

    missing_keys = sorted(valid_keys - set(config))
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise ValueError(f"Missing training config key(s) in {path}: {joined}")

    return config
