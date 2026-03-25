from enum import StrEnum
from typing import Literal

Direction = Literal["right", "left", "down", "up"]

DIRECTION_TO_IDX: dict[Direction, int] = {
    "right": 0,
    "left": 1,
    "down": 2,
    "up": 3,
}

class InferenceMode(StrEnum):
    GAPFILL = "gapfill"
    OUTPAINTING = "outpainting"
    INPAINTING = "inpainting"
    THREE_DIMENSIONAL_IMPUTATION = "three_dimensional_imputation"