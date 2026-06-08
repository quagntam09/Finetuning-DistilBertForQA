from pathlib import Path

import yaml


class Config:
    OPTIONAL_DEFAULTS = {
        "init_checkpoint_dir": None,
        "train_file": None,
        "validation_file": None,
        "test_file": None,
        "cache_dir": None,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_yaml(cls, path="config/model.yaml", profile=None):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

        if "profiles" in data:
            config = data.get("defaults", {}).copy()
            profile_name = profile or data.get("default_profile")
            profiles = data.get("profiles", {})
            if profile_name not in profiles:
                raise ValueError(f"Unknown profile: {profile_name}")
            config.update(profiles[profile_name])
            config["profile_name"] = profile_name
        else:
            config = data

        for key, value in cls.OPTIONAL_DEFAULTS.items():
            config.setdefault(key, value)

        return cls(**config)

    @classmethod
    def available_profiles(cls, path="config/model.yaml"):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return sorted(data.get("profiles", {}))

    @classmethod
    def default_pipeline_profiles(cls, path="config/model.yaml"):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return list(data.get("default_pipeline_profiles") or [])

    def to_yaml(self, path):
        Path(path).write_text(
            yaml.dump(self.__dict__, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
