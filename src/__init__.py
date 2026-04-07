"""
DISCO public API.

All public classes and functions are available directly from this module:

    from src import LatentDiffusionTrainer, InpaintTrainTask
"""

# Autoencoder
from .core.autoencoder import (
    Autoencoder,
    AutoencoderArtifact,
    AutoencoderInferencer,
    AutoencoderTrainer,
    AutoencoderTrainerConfig,
)

# GCNN
from .core.gcnn import (
    GCNClassifier,
    GCNNArtifact,
    GCNNInferencer,
    GCNNTrainer,
    GCNNTrainerConfig,
    RegionGraphDataset,
)

# Latent Diffusion
from .core.latent_diffusion import (
    BaseLatentInferencer,
    GapfillInferencer,
    GapfillRunConfig,
    InpaintInferencer,
    InpaintRunConfig,
    InpaintTrainTask,
    LatentArchitectureSpec,
    LatentBaseRunConfig,
    LatentDiffusionArtifact,
    LatentDiffusionTrainer,
    LatentTrainerConfig,
    LatentTrainResult,
    LatentTrainTask,
    OutpaintInferencer,
    OutpaintRunConfig,
    OutpaintTrainTask,
    ThreeDimImputationInferencer,
    ThreeDimImputationRunConfig,
    ThreeDimImputationTrainTask,
    ThreeDimImputationWeightSweepConfig,
)

# Morphe
from .core.morphe import (
    MorpheArtifact,
    MorpheInferencer,
    MorpheTrainer,
)

# Pixel Diffusion
from .core.pixel_diffusion import (
    Cascade512Evaluator,
    Cascade512Trainer,
    Cascade512TrainerConfig,
    InpaintPrecomputeTask,
    OutpaintPrecomputeTask,
    PixelDatasetPrecomputer,
    PixelDiffusionArtifact,
    PixelDiffusionInferencer,
    PixelPrecomputeTask,
    PrecomputedCascadeDataset,
    ThreeDimImputationPrecomputeTask,
)

# Preprocessor
from .core.preprocessor import (
    PreProcessConfig,
    PreProcessor,
)

# Visualization
from .viz import (
    DEFAULT_CELL_TYPE_COLORS,
    plot_decoded_image,
    plot_inpainting_triplet,
    plot_loss_curve,
    save_region_rgb_images,
    save_side_by_side_barplot,
    show_cell_region_scatterplot,
)


__all__ = [
    # Autoencoder
    "Autoencoder",
    "AutoencoderArtifact",
    "AutoencoderInferencer",
    "AutoencoderTrainer",
    "AutoencoderTrainerConfig",
    # GCNN
    "GCNClassifier",
    "GCNNArtifact",
    "GCNNInferencer",
    "GCNNTrainer",
    "GCNNTrainerConfig",
    "RegionGraphDataset",
    # Latent Diffusion
    "BaseLatentInferencer",
    "GapfillInferencer",
    "GapfillRunConfig",
    "InpaintInferencer",
    "InpaintRunConfig",
    "InpaintTrainTask",
    "LatentArchitectureSpec",
    "LatentBaseRunConfig",
    "LatentDiffusionArtifact",
    "LatentDiffusionTrainer",
    "LatentTrainerConfig",
    "LatentTrainResult",
    "LatentTrainTask",
    "OutpaintInferencer",
    "OutpaintRunConfig",
    "OutpaintTrainTask",
    "ThreeDimImputationInferencer",
    "ThreeDimImputationRunConfig",
    "ThreeDimImputationTrainTask",
    "ThreeDimImputationWeightSweepConfig",
    # Morphe
    "MorpheArtifact",
    "MorpheInferencer",
    "MorpheTrainer",
    # Pixel Diffusion
    "Cascade512Evaluator",
    "Cascade512Trainer",
    "Cascade512TrainerConfig",
    "InpaintPrecomputeTask",
    "OutpaintPrecomputeTask",
    "PixelDatasetPrecomputer",
    "PixelDiffusionArtifact",
    "PixelDiffusionInferencer",
    "PixelPrecomputeTask",
    "PrecomputedCascadeDataset",
    "ThreeDimImputationPrecomputeTask",
    # Preprocessor
    "PreProcessConfig",
    "PreProcessor",
    # Visualization
    "DEFAULT_CELL_TYPE_COLORS",
    "plot_decoded_image",
    "plot_inpainting_triplet",
    "plot_loss_curve",
    "save_region_rgb_images",
    "save_side_by_side_barplot",
    "show_cell_region_scatterplot",
]
