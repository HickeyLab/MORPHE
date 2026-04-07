from typing import Any

import torch


def get_config_attr(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)

def set_config_attr(config: Any, field: str, value: str) -> Any:
    if isinstance(config, dict):
        config[field] = value
    else:
        setattr(config, field, value)
    return config

def resolve_device(device: torch.device | str | None) -> torch.device:
    """Resolve a user-provided device or choose a sensible default."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def resolve_dtype(device: torch.device, dtype: torch.dtype | None) -> torch.dtype:
    """Resolve an inference dtype based on device when none is provided."""
    if dtype is not None:
        return dtype
    return torch.float16 if device.type == "cuda" else torch.float32