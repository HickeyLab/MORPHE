from typing import Type

from core.latent_diffusion.inference.base import BaseLatentInferencer
from core.latent_diffusion.inference.gapfill import GapfillInferencer
from core.latent_diffusion.inference.inpaint import InpaintInferencer
from core.latent_diffusion.inference.outpaint import OutpaintInferencer
from core.latent_diffusion.inference.three_dim import ThreeDimImputationInferencer
from constants import InferenceMode


_LATENT_INFERENCER_REGISTRY: dict[
    InferenceMode,
    Type[BaseLatentInferencer],
] = {
    InferenceMode.GAPFILL: GapfillInferencer,
    InferenceMode.INPAINTING: InpaintInferencer,
    InferenceMode.OUTPAINTING: OutpaintInferencer,
    InferenceMode.THREE_DIMENSIONAL_IMPUTATION: ThreeDimImputationInferencer,
}