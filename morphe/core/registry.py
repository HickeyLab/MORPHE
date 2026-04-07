from enum import StrEnum

from ..constants import InferenceMode
from .latent_diffusion.inference.base import BaseLatentInferencer
from .latent_diffusion.inference.gapfill import GapfillInferencer
from .latent_diffusion.inference.inpaint import InpaintInferencer
from .latent_diffusion.inference.outpaint import OutpaintInferencer
from .latent_diffusion.inference.three_dim import ThreeDimImputationInferencer
from .latent_diffusion.train.base import LatentTrainTask
from .latent_diffusion.train.inpaint import InpaintTrainTask
from .latent_diffusion.train.outpaint import OutpaintTrainTask
from .latent_diffusion.train.three_dim import ThreeDimImputationTrainTask
from .pixel_diffusion.precompute.inpaint import InpaintPrecomputeTask
from .pixel_diffusion.precompute.outpaint import OutpaintPrecomputeTask
from .pixel_diffusion.precompute.three_dim import ThreeDimImputationPrecomputeTask
    
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