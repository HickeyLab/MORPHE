from typing import Type

from disco.config import InferenceMode
from disco.core.latent_diffusion.infer.base import BaseLatentInferencer
from disco.core.latent_diffusion.infer.gapfill import GapfillInferencer
from disco.core.latent_diffusion.infer.inpaint import InpaintInferencer
from disco.core.latent_diffusion.infer.outpaint import OutpaintInferencer
from disco.core.latent_diffusion.infer.three_dim import ThreeDimImputationInferencer


_LATENT_INFERENCER_REGISTRY: dict[
    InferenceMode,
    Type[BaseLatentInferencer],
] = {
    InferenceMode.GAPFILL: GapfillInferencer,
    InferenceMode.INPAINTING: InpaintInferencer,
    InferenceMode.OUTPAINTING: OutpaintInferencer,
    InferenceMode.THREE_DIMENSIONAL_IMPUTATION: ThreeDimImputationInferencer,
}