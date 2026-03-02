from dataclasses import dataclass


@dataclass(frozen=True)
class DISCOConfig:
    model_name: str
    original_dimensions: tuple[int, int] = (9406, 9070)