from dataclasses import dataclass
from enum import StrEnum

from disco.core.latent_diffusion.infer.base import BaseLatentInferencer
from disco.core.latent_diffusion.infer.gapfill import GapfillInferencer
from disco.core.latent_diffusion.infer.inpaint import InpaintInferencer
from disco.core.latent_diffusion.infer.outpaint import OutpaintInferencer
from disco.core.latent_diffusion.infer.three_dim import ThreeDimImputationInferencer
from disco.core.latent_diffusion.train.base import LatentTrainStrategy
from disco.core.latent_diffusion.train.inpaint import InpaintTrainStrategy
from disco.core.latent_diffusion.train.outpaint import OutpaintTrainStrategy
from disco.core.latent_diffusion.train.three_dim import ThreeDimImputationTrainStrategy
from disco.core.pixel_diffusion.precompute.inpaint import InpaintPrecomputeStrategy
from disco.core.pixel_diffusion.precompute.outpaint import OutpaintPrecomputeStrategy
from disco.core.pixel_diffusion.precompute.three_dim import ThreeDimImputationPrecomputeStrategy


@dataclass(frozen=True)
class PreProcessConfig:
    original_dimensions: tuple[int, int] = (9406, 9070)
    
class InferenceMode(StrEnum):
    GAPFILL = "gapfill"
    OUTPAINTING = "outpainting"
    INPAINTING = "inpainting"
    THREE_DIMENSIONAL_IMPUTATION = "three_dimensional_imputation"
    
LATENT_INFERENCER_REGISTRY: dict[InferenceMode, type[BaseLatentInferencer]] = {
    InferenceMode.GAPFILL: GapfillInferencer,
    InferenceMode.INPAINTING: InpaintInferencer,
    InferenceMode.OUTPAINTING: OutpaintInferencer,
    InferenceMode.THREE_DIMENSIONAL_IMPUTATION: ThreeDimImputationInferencer,
}

LATENT_TRAIN_STRATEGY_REGISTRY: dict[InferenceMode, type[LatentTrainStrategy]] = {
    InferenceMode.INPAINTING: InpaintTrainStrategy,
    InferenceMode.GAPFILL: OutpaintTrainStrategy,
    InferenceMode.OUTPAINTING: OutpaintTrainStrategy,
    InferenceMode.THREE_DIMENSIONAL_IMPUTATION: ThreeDimImputationTrainStrategy,
}

PRECOMPUTE_STRATEGY_REGISTRY = {
    InferenceMode.INPAINTING: InpaintPrecomputeStrategy,
    InferenceMode.GAPFILL: OutpaintPrecomputeStrategy,
    InferenceMode.OUTPAINTING: OutpaintPrecomputeStrategy,
    InferenceMode.THREE_DIMENSIONAL_IMPUTATION: ThreeDimImputationPrecomputeStrategy,
}