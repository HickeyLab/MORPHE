from disco.config import InferenceMode
from disco.core.pixel_diffusion.precompute.inpaint import InpaintPrecomputeStrategy
from disco.core.pixel_diffusion.precompute.outpaint import OutpaintPrecomputeStrategy
from disco.core.pixel_diffusion.precompute.three_dim import ThreeDimImputationPrecomputeStrategy


EXPECTED_PRECOMPUTE_STRATEGY_BY_MODE = {
    InferenceMode.INPAINTING: InpaintPrecomputeStrategy,
    InferenceMode.GAPFILL: OutpaintPrecomputeStrategy,
    InferenceMode.OUTPAINTING: OutpaintPrecomputeStrategy,
    InferenceMode.THREE_DIMENSIONAL_IMPUTATION: ThreeDimImputationPrecomputeStrategy,
}