from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ..autoencoder.artifact import AutoencoderArtifact
from ..autoencoder.inferencer import AutoencoderInferencer
from ..latent_diffusion.inference.base import BaseLatentInferencer
from ..latent_diffusion.inference.gapfill import GapfillInferencer
from ..latent_diffusion.inference.inpaint import InpaintInferencer
from ..latent_diffusion.inference.outpaint import OutpaintInferencer
from ..latent_diffusion.inference.three_dim import ThreeDimImputationInferencer
from .artifact import MorpheArtifact
from ..gcnn.artifact import GCNNArtifact
from ..gcnn.inferencer import GCNNInferencer
from ..latent_diffusion.artifact import LatentDiffusionArtifact
from ..pixel_diffusion.artifact import PixelDiffusionArtifact
from ..pixel_diffusion.inferencer import PixelDiffusionInferencer
from ..preprocessor.config import PreProcessConfig
from ..preprocessor.preprocess import PreProcessor
from ..registry import LATENT_INFERENCER_REGISTRY
from ...utils import resolve_device


class MorpheInferencer:
    """
    Top-level inference facade for MORPHE pipelines.

    This class owns the resolved runtime components built from a
    ``MorpheArtifact`` and exposes higher-level prediction / generation
    methods across the full MORPHE stack.
    """

    def __init__(
        self,
        *,
        artifact: MorpheArtifact,
        device: torch.device,
        preprocessor: PreProcessor,
        gcnn_inferencer: GCNNInferencer,
        autoencoder_inferencer: AutoencoderInferencer,
        ld_inferencer: BaseLatentInferencer,
        pd_inferencer: PixelDiffusionInferencer,
    ) -> None:
        """
        Initialize a fully constructed MORPHE inferencer.

        Args:
            artifact: Top-level artifact containing the trained MORPHE
                component artifacts.
            device: Resolved runtime device for inference.
            dtype: Resolved floating-point dtype for inference, when used.
            gcnn_inferencer: Built GCNN inferencer, if available.
            autoencoder_inferencer: Built autoencoder inferencer, if available.
            ld_inferencer: Built latent diffusion inferencer, if available.
            pd_inferencer: Built pixel diffusion inferencer, if available.
        """
        if artifact is None or not isinstance(artifact, MorpheArtifact):
            raise ValueError("artifact must be a valid MorpheArtifact instance.")
        if preprocessor is None or not isinstance(preprocessor, PreProcessor):
            raise ValueError("preprocessor must be a valid PreProcessor instance.")
        if gcnn_inferencer is None or not isinstance(gcnn_inferencer, GCNNInferencer):
            raise ValueError("gcnn_inferencer must be a valid GCNNInferencer instance.")
        if autoencoder_inferencer is None or not isinstance(
            autoencoder_inferencer, AutoencoderInferencer
        ):
            raise ValueError(
                "autoencoder_inferencer must be a valid AutoencoderInferencer instance."
            )
        if ld_inferencer is None or not isinstance(ld_inferencer, BaseLatentInferencer):
            raise ValueError(
                "ld_inferencer must be a valid BaseLatentInferencer instance."
            )
        if pd_inferencer is None or not isinstance(
            pd_inferencer, PixelDiffusionInferencer
        ):
            raise ValueError(
                "pd_inferencer must be a valid PixelDiffusionInferencer instance."
            )

        self.artifact = artifact
        self.device = device
        self.dtype = torch.float32
        self.preprocessor = preprocessor
        self.gcnn_inferencer = gcnn_inferencer
        self.autoencoder_inferencer = autoencoder_inferencer
        self.latent_diffusion_inferencer = ld_inferencer
        self.pixel_diffusion_inferencer = pd_inferencer

    @classmethod
    def from_artifact(
        cls,
        *,
        artifact: MorpheArtifact,
        device: torch.device | str | None = None,
    ) -> "MorpheInferencer":
        """
        Build a ``MorpheInferencer`` from a top-level artifact.

        Args:
            artifact: Top-level MORPHE artifact.
            device: Target device for inference.

        Returns:
            A fully initialized ``MorpheInferencer``.
        """
        resolved_device = resolve_device(device)

        if (
            artifact.preprocessor_config is None
            or not isinstance(artifact.preprocessor_config, PreProcessConfig)
            or artifact.gcnn_artifact is None
            or not isinstance(artifact.gcnn_artifact, GCNNArtifact)
            or artifact.autoencoder_artifact is None
            or not isinstance(artifact.autoencoder_artifact, AutoencoderArtifact)
            or artifact.latent_diffusion_artifact is None
            or not isinstance(
                artifact.latent_diffusion_artifact, LatentDiffusionArtifact
            )
            or artifact.pixel_diffusion_artifact is None
            or not isinstance(artifact.pixel_diffusion_artifact, PixelDiffusionArtifact)
        ):
            raise ValueError(
                "Autoencoder, GCNN, Latent Diffusion, and Pixel Diffusion artifacts must be present to build an inferencer."
            )

        preprocessor = PreProcessor(config=artifact.preprocessor_config)
        gcnn_inferencer = GCNNInferencer.from_artifact(
            artifact=artifact.gcnn_artifact,
            device=resolved_device,
        )
        autoencoder_inferencer = AutoencoderInferencer.from_artifact(
            artifact=artifact.autoencoder_artifact,
            device=resolved_device,
        )

        inference_cls = LATENT_INFERENCER_REGISTRY.get(
            artifact.latent_diffusion_artifact.inference_mode
        )
        if not inference_cls:
            raise ValueError(
                f"Unsupported inference mode {artifact.latent_diffusion_artifact.inference_mode} in latent diffusion artifact."
            )

        ld_inferencer = inference_cls.from_artifact(
            artifact=artifact.latent_diffusion_artifact,
            device=resolved_device,
        )
        pd_inferencer = PixelDiffusionInferencer.from_artifact(
            artifact=artifact.pixel_diffusion_artifact,
            device=resolved_device,
        )

        return cls(
            artifact=artifact,
            device=resolved_device,
            preprocessor=preprocessor,
            gcnn_inferencer=gcnn_inferencer,
            autoencoder_inferencer=autoencoder_inferencer,
            ld_inferencer=ld_inferencer,
            pd_inferencer=pd_inferencer,
        )

    def _prepare_regions(
        self,
        df: pd.DataFrame,
        root_dir: str | Path,
    ) -> None:
        """
        Run the shared preprocessing, GCNN prediction, and autoencoder
        rasterization pipeline needed before latent diffusion inference.

        Args:
            df: Input dataframe containing raw per-cell or per-spot data.
            root_dir: Directory where intermediate rasterized region outputs
                should be written.
        """
        if not root_dir:
            raise ValueError("Must provide root_dir for saving intermediate files.")

        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        processed_df, _ = self.preprocessor.preprocess(df=df)
        gcnn_probabilities = self.gcnn_inferencer.predict_proba(df=processed_df)

        self.autoencoder_inferencer.add_rgb_and_rasterize_per_region(
            result_df=gcnn_probabilities,
            save_dir=root_dir,
            split=False,
            region_col=self.artifact.region_col,
            x_col=self.artifact.pos_cols[0],
            y_col=self.artifact.pos_cols[1],
        )

    def _decode_latents_to_cell_maps(
        self,
        latents: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Decode latent outputs into final cell-map predictions.

        Each latent is first decoded into RGB space with the pixel diffusion
        inferencer, then converted from RGB into a cell-map representation
        with the autoencoder inferencer.

        Args:
            latents: Latent tensors returned by a latent diffusion inferencer.

        Returns:
            Decoded cell-map tensors in the same order as the input latents.
        """
        results: list[torch.Tensor] = []

        for latent in latents:
            rgb = self.pixel_diffusion_inferencer.decode(latent=latent)
            cell_map = self.autoencoder_inferencer.decode_rgb_to_cell_map(rgb)
            results.append(cell_map)

        return results

    def run_gapfill(
        self,
        df: pd.DataFrame,
        *,
        root_dir: str | Path,
        save_dir: str | Path,
        save_name: str = "default",
        num_steps: int = 200,
        iterations: int = 10,
        show_plot: bool = False,
        plot_title: str | None = None,
        plot_fig_size: tuple[int, int] = (6, 6),
        bbox: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Run the full MORPHE pipeline for latent gap-filling.

        Args:
            df: Input dataframe containing raw region data.
            root_dir: Directory for intermediate rasterized outputs.
            save_dir: Directory where latent gap-fill outputs should be saved.
            save_name: Base name used for saved outputs.
            steps: Number of diffusion denoising steps per iteration.
            iterations: Number of iterative gap-fill passes to run.
            show_plot: Whether to display qualitative plots during inference.
            plot_title: Optional plot title override.
            plot_fig_size: Figure size used for visualization.
            bbox: Optional bounding box conditioning tensor.

        Returns:
            Final decoded cell-map tensors.
        """
        self._prepare_regions(df=df, root_dir=root_dir)

        if not isinstance(self.latent_diffusion_inferencer, GapfillInferencer):
            raise TypeError(
                "latent_diffusion_inferencer must be a GapfillInferencer "
                "to use run_gapfill()."
            )

        latents = self.latent_diffusion_inferencer.run(
            input_dir=root_dir,
            save_dir=save_dir,
            save_name=save_name,
            num_steps=num_steps,
            iterations=iterations,
            show_plot=show_plot,
            plot_title=plot_title,
            plot_fig_size=plot_fig_size,
            bbox=bbox,
        )
        return self._decode_latents_to_cell_maps(latents)

    def run_inpaint(
        self,
        df: pd.DataFrame,
        *,
        root_dir: str | Path,
        mask_dir: str | Path,
        num_steps: int = 200,
        show_plot: bool = False,
        plot_title: str = "Inpainting Progress",
        plot_fig_size: tuple[int, int] = (6, 6),
    ) -> list[torch.Tensor]:
        """
        Run the full MORPHE pipeline for latent inpainting.

        Args:
            df: Input dataframe containing raw region data.
            root_dir: Directory containing rasterized inputs for inpainting.
            mask_dir: Directory containing inpainting masks.
            num_steps: Number of diffusion denoising steps.
            show_plot: Whether to display qualitative plots during inference.
            plot_title: Title used for qualitative plots.
            plot_fig_size: Figure size used for visualization.

        Returns:
            Final decoded cell-map tensors.
        """
        self._prepare_regions(df=df, root_dir=root_dir)

        if not isinstance(self.latent_diffusion_inferencer, InpaintInferencer):
            raise TypeError(
                "latent_diffusion_inferencer must be an InpaintInferencer "
                "to use run_inpaint()."
            )

        latents = self.latent_diffusion_inferencer.run(
            input_dir=root_dir,
            mask_dir=mask_dir,
            num_steps=num_steps,
            show_plot=show_plot,
            plot_title=plot_title,
            plot_fig_size=plot_fig_size,
        )
        return self._decode_latents_to_cell_maps(latents)

    def run_outpaint(
        self,
        df: pd.DataFrame,
        *,
        root_dir: str | Path,
        save_dir: str | Path,
        save_name: str = "default",
        num_steps: int = 200,
        crop_ratio: float = 0.97,
        iterations: int = 10,
        direction: str = "right",
        show_plot: bool = False,
        plot_title: str | None = None,
        plot_fig_size: tuple[int, int] = (6, 6),
    ) -> list[torch.Tensor]:
        """
        Run the full MORPHE pipeline for latent outpainting.

        Args:
            df: Input dataframe containing raw region data.
            root_dir: Directory for intermediate rasterized outputs.
            input_dir: Directory containing rasterized inputs for outpainting.
            save_dir: Directory where outpainting outputs should be saved.
            save_name: Base name used for saved outputs.
            num_steps: Number of diffusion denoising steps per iteration.
            crop_ratio: Fraction of the image to preserve per iteration.
            iterations: Number of iterative outpainting passes to run.
            direction: Direction to expand ("right", "left", "up", or "down").
            show_plot: Whether to display qualitative plots during inference.
            plot_title: Optional plot title override.
            plot_fig_size: Figure size used for visualization.

        Returns:
            Final decoded cell-map tensors.
        """
        self._prepare_regions(df=df, root_dir=root_dir)

        if not isinstance(self.latent_diffusion_inferencer, OutpaintInferencer):
            raise TypeError(
                "latent_diffusion_inferencer must be an OutpaintInferencer "
                "to use run_outpaint()."
            )

        latents = self.latent_diffusion_inferencer.run(
            input_dir=root_dir,
            save_dir=save_dir,
            save_name=save_name,
            num_steps=num_steps,
            crop_ratio=crop_ratio,
            iterations=iterations,
            direction=direction,
            show_plot=show_plot,
            plot_title=plot_title,
            plot_fig_size=plot_fig_size,
        )
        return self._decode_latents_to_cell_maps(latents)

    def run_three_dim_imputation(
        self,
        df: pd.DataFrame,
        root_dir: str | Path,
        *,
        prev_path: str | Path,
        next_path: str | Path,
        save_dir: str | Path,
        save_name: str = "default",
        num_steps: int = 200,
        w_prev: float = 0.5,
        w_next: float = 0.5,
        save_latents_name: str = "latents.pt",
        save_png_name: str = "output.png",
    ) -> list[torch.Tensor]:
        """
        Run the full MORPHE pipeline for 3D latent imputation.

        Args:
            df: Input dataframe containing raw region data.
            root_dir: Directory for intermediate rasterized outputs.
            save_dir: Directory for intermediate rasterized outputs.
            prev_path: Path to the previous slice input image.
            next_path: Path to the next slice input image.
            out_dir: Output directory for imputation results.
            num_steps: Number of diffusion denoising steps.
            w_prev: Weight applied to the previous slice conditioning.
            w_next: Weight applied to the next slice conditioning.
            save_latents_name: Filename used when saving latent outputs.
            save_png_name: Filename used when saving decoded image outputs.

        Returns:
            Final decoded cell-map tensors.
        """
        self._prepare_regions(df=df, root_dir=root_dir)

        if not isinstance(
            self.latent_diffusion_inferencer,
            ThreeDimImputationInferencer,
        ):
            raise TypeError(
                "latent_diffusion_inferencer must be a "
                "ThreeDimImputationInferencer to use "
                "run_three_dim_imputation()."
            )

        latent = self.latent_diffusion_inferencer.run(
            prev_img_path=prev_path,
            next_img_path=next_path,
            save_dir=save_dir,
            save_name=save_name,
            num_steps=num_steps,
            w_prev=w_prev,
            w_next=w_next,
            save_latents_name=save_latents_name,
            save_png_name=save_png_name,
        )
        return self._decode_latents_to_cell_maps([latent])
