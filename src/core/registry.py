from enum import StrEnum

from constants import InferenceMode
from core.latent_diffusion.inference.base import BaseLatentInferencer
from core.latent_diffusion.inference.gapfill import GapfillInferencer
from core.latent_diffusion.inference.inpaint import InpaintInferencer
from core.latent_diffusion.inference.outpaint import OutpaintInferencer
from core.latent_diffusion.inference.three_dim import ThreeDimImputationInferencer
from core.latent_diffusion.train.base import LatentTrainStrategy
from core.latent_diffusion.train.inpaint import InpaintTrainStrategy
from core.latent_diffusion.train.outpaint import OutpaintTrainStrategy
from core.latent_diffusion.train.three_dim import ThreeDimImputationTrainStrategy
from core.pixel_diffusion.precompute.inpaint import InpaintPrecomputeStrategy
from core.pixel_diffusion.precompute.outpaint import OutpaintPrecomputeStrategy
from core.pixel_diffusion.precompute.three_dim import ThreeDimImputationPrecomputeStrategy
    
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