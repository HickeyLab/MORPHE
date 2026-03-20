from typing import Any


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