from enum import StrEnum

from constants import InferenceMode
from core.latent_diffusion.inference.base import BaseLatentInferencer
from core.latent_diffusion.inference.gapfill import GapfillInferencer
from core.latent_diffusion.inference.inpaint import InpaintInferencer
from core.latent_diffusion.inference.outpaint import OutpaintInferencer
from core.latent_diffusion.inference.three_dim import ThreeDimImputationInferencer
from core.latent_diffusion.train.base import LatentTrainTask
from core.latent_diffusion.train.inpaint import InpaintTrainTask
from core.latent_diffusion.train.outpaint import OutpaintTrainTask
from core.latent_diffusion.train.three_dim import ThreeDimImputationTrainTask
from core.pixel_diffusion.precompute.inpaint import InpaintPrecomputeTask
from core.pixel_diffusion.precompute.outpaint import OutpaintPrecomputeTask
from core.pixel_diffusion.precompute.three_dim import ThreeDimImputationPrecomputeTask
    
LATENT_INFERENCER_REGISTRY: dict[InferenceMode, type[BaseLatentInferencer]] = {
    InferenceMode.GAPFILL: GapfillInferencer,
    InferenceMode.INPAINTING: InpaintInferencer,
    InferenceMode.OUTPAINTING: OutpaintInferencer,
    InferenceMode.THREE_DIMENSIONAL_IMPUTATION: ThreeDimImputationInferencer,
}

LATENT_TRAIN_TASK_REGISTRY: dict[InferenceMode, type[LatentTrainTask]] = {
    InferenceMode.INPAINTING: InpaintTrainTask,
    InferenceMode.GAPFILL: OutpaintTrainTask,
    InferenceMode.OUTPAINTING: OutpaintTrainTask,
    InferenceMode.THREE_DIMENSIONAL_IMPUTATION: ThreeDimImputationTrainTask,
}

PRECOMPUTE_TASK_REGISTRY = {
    InferenceMode.INPAINTING: InpaintPrecomputeTask,
    InferenceMode.GAPFILL: OutpaintPrecomputeTask,
    InferenceMode.OUTPAINTING: OutpaintPrecomputeTask,
    InferenceMode.THREE_DIMENSIONAL_IMPUTATION: ThreeDimImputationPrecomputeTask,
}