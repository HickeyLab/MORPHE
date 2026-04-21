from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .artifact import AutoencoderArtifact
from .model import Autoencoder
from .utils.rasterize import rasterize_rgb_regions
from ...utils import resolve_device, resolve_dtype


class AutoencoderInferencer:
    """
    Inference wrapper for converting between autoencoder inputs, RGB encodings,
    decoder logits, and rasterized region outputs.

    This class rebuilds a trained autoencoder from an artifact and provides
    helper methods for:
    - encoding feature matrices to RGB triplets,
    - adding RGB columns to dataframes,
    - decoding RGB inputs back to logits,
    - converting decoded logits into cell maps,
    - rasterizing per-region RGB outputs.
    """

    def __init__(
        self,
        artifact: AutoencoderArtifact,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize an inferencer from a serialized autoencoder artifact.

        Args:
            artifact: Serialized artifact containing model weights and metadata
                required for inference.
            device: Target device for model execution. If ``None``, a default
                device is resolved automatically.
            dtype: Target data type for model execution. If ``None``, a default
                dtype is resolved based on the target device.
        """
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)

        self.artifact = artifact
        self.model: Autoencoder = artifact.build_model(
            device=self.device, dtype=self.dtype
        )

    @classmethod
    def from_artifact(
        cls,
        artifact: AutoencoderArtifact,
        *,
        device: torch.device | str | None = None,
    ) -> AutoencoderInferencer:
        """
        Construct an inferencer from an autoencoder artifact.

        Args:
            artifact: Serialized artifact containing model weights and metadata
                required for inference.
            device: Target device for model execution. If ``None``, a default
                device is resolved automatically.

        Returns:
            An initialized ``AutoencoderInferencer``.

        Raises:
            TypeError: If ``artifact`` is not an ``AutoencoderArtifact``.
        """
        if not isinstance(artifact, AutoencoderArtifact):
            raise TypeError(
                "Expected artifact to be an instance of AutoencoderArtifact."
            )

        return cls(
            artifact=artifact,
            device=device,
            dtype=torch.float32,
        )

    def _to_embedding_tensor(
        self,
        emb_matrix: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """
        Convert an embedding matrix input to a torch tensor.

        Args:
            emb_matrix: Input embedding matrix as either a tensor or NumPy array.

        Returns:
            The input converted to a torch tensor.

        Raises:
            TypeError: If ``emb_matrix`` is not a tensor or NumPy array.
        """
        if isinstance(emb_matrix, np.ndarray):
            emb_matrix = torch.from_numpy(emb_matrix)

        if not isinstance(emb_matrix, torch.Tensor):
            raise TypeError("emb_matrix must be a torch.Tensor or numpy.ndarray.")

        return emb_matrix

    def _validate_embedding_matrix(
        self,
        emb_matrix: torch.Tensor,
    ) -> None:
        """
        Validate the shape of an embedding matrix used for encoder inference.

        Args:
            emb_matrix: Embedding matrix expected to have shape
                ``(n_samples, in_dim)``.

        Raises:
            ValueError: If the tensor is not 2D or does not match the expected
                feature dimension.
        """
        if emb_matrix.ndim != 2:
            raise ValueError(
                f"emb_matrix must have shape (n, {self.artifact.in_dim}), "
                f"got {tuple(emb_matrix.shape)}."
            )

        if emb_matrix.shape[1] != self.artifact.in_dim:
            raise ValueError(
                f"emb_matrix must have {self.artifact.in_dim} features, "
                f"got {emb_matrix.shape[1]}."
            )

    def _validate_dataframe_columns(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate that a dataframe contains all required model input columns.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If one or more required input columns are missing.
        """
        missing = [col for col in self.artifact.input_cols if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required input columns: {missing}")

    def _validate_rgb_output_column_names(
        self,
        *,
        r_col: str,
        g_col: str,
        b_col: str,
    ) -> None:
        """
        Validate that output RGB column names are distinct.

        Args:
            r_col: Name of the red output column.
            g_col: Name of the green output column.
            b_col: Name of the blue output column.

        Raises:
            ValueError: If any output column names are duplicated.
        """
        if len({r_col, g_col, b_col}) != 3:
            raise ValueError("r_col, g_col, and b_col must be distinct.")

    def _validate_rgb_tensor(
        self,
        rgb: torch.Tensor,
    ) -> None:
        """
        Validate an RGB tensor accepted by decoder helpers.

        Accepted shapes are:
        - ``(n, 3)`` for a batch of RGB triplets
        - ``(3, h, w)`` for a channel-first RGB image

        Args:
            rgb: RGB tensor to validate.

        Raises:
            TypeError: If ``rgb`` is not a torch tensor.
            ValueError: If ``rgb`` does not have a supported shape.
        """
        if not isinstance(rgb, torch.Tensor):
            raise TypeError("rgb must be a torch.Tensor.")

        if rgb.ndim not in (2, 3):
            raise ValueError(
                f"rgb must have ndim 2 or 3, got shape {tuple(rgb.shape)}."
            )

        if rgb.ndim == 2 and rgb.shape[1] != 3:
            raise ValueError(f"2D rgb must have shape (n, 3), got {tuple(rgb.shape)}.")

        if rgb.ndim == 3 and rgb.shape[0] != 3:
            raise ValueError(
                f"3D rgb must have shape (3, h, w), got {tuple(rgb.shape)}."
            )

    def _validate_rgb_image(
        self,
        rgb: torch.Tensor,
    ) -> None:
        """
        Validate a channel-first RGB image tensor.

        Args:
            rgb: RGB image tensor expected to have shape ``(3, h, w)``.

        Raises:
            TypeError: If ``rgb`` is not a torch tensor.
            ValueError: If ``rgb`` does not have shape ``(3, h, w)``.
        """
        if not isinstance(rgb, torch.Tensor):
            raise TypeError("rgb must be a torch.Tensor.")

        if rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError(f"rgb must have shape (3, h, w), got {tuple(rgb.shape)}.")
        
    def _validate_required_dataframe_columns(
        self,
        df: pd.DataFrame,
        *,
        columns: Sequence[str],
        context: str,
    ) -> None:
        """
        Validate that a dataframe contains all required columns for a given context.
        """
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required {context} columns: {missing}"
            )


    def _validate_distinct_column_names(
        self,
        *,
        columns: Sequence[str],
        names: Sequence[str],
        context: str,
    ) -> None:
        """
        Validate that the provided column names are all distinct.
        """
        if len(columns) != len(set(columns)):
            duplicates = sorted({col for col in columns if columns.count(col) > 1})
            raise ValueError(
                f"{context} column names must be distinct. Duplicates: {duplicates}"
            )


    def _validate_rasterization_params(
        self,
        *,
        val_ratio: float,
        image_size: int,
        file_name_prefix: str,
    ) -> None:
        """
        Validate scalar parameters used for rasterization.
        """
        if not isinstance(val_ratio, (int, float)):
            raise TypeError("val_ratio must be a float.")
        if not 0.0 <= float(val_ratio) <= 1.0:
            raise ValueError(f"val_ratio must be in [0, 1], got {val_ratio}.")

        if not isinstance(image_size, int):
            raise TypeError("image_size must be an int.")
        if image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {image_size}.")

        if not isinstance(file_name_prefix, str):
            raise TypeError("file_name_prefix must be a string.")
        if not file_name_prefix.strip():
            raise ValueError("file_name_prefix must be a non-empty string.")

    def _encode_to_rgb(
        self,
        emb_matrix: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """
        Encode an input embedding matrix into RGB triplets.

        Args:
            emb_matrix: Input matrix of shape ``(n_samples, in_dim)`` as either
                a torch tensor or NumPy array.

        Returns:
            A CPU uint8 tensor of shape ``(n_samples, 3)`` containing RGB values.
        """
        emb_matrix = self._to_embedding_tensor(emb_matrix)
        self._validate_embedding_matrix(emb_matrix)

        z_min, z_max = self.artifact.z_min, self.artifact.z_max

        with torch.no_grad():
            z = self.model.encoder(emb_matrix.to(self.device, dtype=self.dtype))

            z_min = z_min.to(self.device)
            z_max = z_max.to(self.device)

            range_vals = z_max - z_min
            range_vals[range_vals == 0] = 1e-9

            scaled_3d = (z - z_min) / range_vals
            scaled_3d = torch.clamp(scaled_3d, 0.0, 1.0)

            rgb_3d = (scaled_3d * 255).round().to(torch.uint8)

        return rgb_3d.cpu()

    def _encode_to_rgb_df(
        self,
        df: pd.DataFrame,
        *,
        r_col: str = "R",
        g_col: str = "G",
        b_col: str = "B",
    ) -> pd.DataFrame:
        """
        Encode dataframe feature columns into RGB columns.

        Args:
            df: Input dataframe containing all columns listed in
                ``artifact.input_cols``.
            r_col: Name of the red output column.
            g_col: Name of the green output column.
            b_col: Name of the blue output column.

        Returns:
            A copy of the input dataframe with added RGB columns.

        Raises:
            ValueError: If required input columns are missing or output RGB
                column names are not distinct.
        """
        self._validate_dataframe_columns(df)
        self._validate_rgb_output_column_names(
            r_col=r_col,
            g_col=g_col,
            b_col=b_col,
        )

        emb_matrix = df.loc[:, self.artifact.input_cols].to_numpy(
            dtype=np.float32, copy=True
        )
        rgb_3d = self._encode_to_rgb(emb_matrix)

        out = df.copy()
        out[r_col] = rgb_3d[:, 0].numpy()
        out[g_col] = rgb_3d[:, 1].numpy()
        out[b_col] = rgb_3d[:, 2].numpy()
        return out

    def decode_rgb_to_logits(
        self,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode RGB inputs back into autoencoder decoder logits.

        Args:
            rgb: RGB input tensor with shape ``(n, 3)`` or ``(3, h, w)``.

        Returns:
            If ``rgb`` has shape ``(n, 3)``, returns logits with shape
            ``(n, in_dim)``.
            If ``rgb`` has shape ``(3, h, w)``, returns logits with shape
            ``(h, w, in_dim)``.
        """
        self._validate_rgb_tensor(rgb)

        z_min, z_max = self.artifact.z_min, self.artifact.z_max

        rgb = rgb.to(self.device, dtype=torch.float32)
        z_min = z_min.to(self.device)
        z_max = z_max.to(self.device)

        range_vals = z_max - z_min
        range_vals[range_vals == 0] = 1e-9

        with torch.no_grad():
            if rgb.ndim == 2:
                z_recovered = rgb * range_vals + z_min
                logits = self.model.decoder(z_recovered)
                return logits

            if rgb.ndim == 3:
                flat_img = rgb.permute(1, 2, 0).reshape(-1, 3)
                white_mask = (flat_img == 1.0).all(dim=1)

                logits = torch.zeros(
                    flat_img.shape[0],
                    self.artifact.in_dim,
                    device=self.device,
                )

                infer_input_rgb = flat_img[~white_mask]
                if infer_input_rgb.shape[0] > 0:
                    z_recovered = infer_input_rgb * range_vals + z_min
                    logits_valid = self.model.decoder(z_recovered)
                    logits[~white_mask] = logits_valid

                h, w = rgb.shape[1], rgb.shape[2]
                return logits.reshape(h, w, self.artifact.in_dim)

        raise ValueError(f"Expected rgb to have ndim 2 or 3, got shape {rgb.shape}.")

    def decode_rgb_to_cell_map(
        self,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode a channel-first RGB image into a cell-map prediction tensor.

        Args:
            rgb: RGB image tensor with shape ``(3, h, w)``.

        Returns:
            A tensor of shape ``(1, h, w)`` containing predicted cell labels,
            where background or masked-out pixels remain zero.
        """
        self._validate_rgb_image(rgb)

        rgb = rgb.to(self.device, dtype=torch.float32)
        h, w = rgb.shape[1], rgb.shape[2]

        flat_img = rgb.permute(1, 2, 0).reshape(-1, 3)
        white_mask = (flat_img == 1.0).all(dim=1)

        pred = torch.zeros(flat_img.shape[0], dtype=torch.long, device=self.device)

        logits = self.decode_rgb_to_logits(rgb)
        logits = logits.reshape(-1, self.artifact.in_dim)

        if (~white_mask).any():
            pred[~white_mask] = torch.argmax(logits[~white_mask], dim=1) + 1

        pred = pred.reshape(1, h, w)
        return pred

    def add_rgb_and_rasterize_per_region(
        self,
        result_df: pd.DataFrame,
        save_dir: str | Path,
        split: bool = True,
        val_ratio: float = 0.2,
        image_size: int = 512,
        file_name_prefix: str = "region",
        seed: int = 42,
        x_col: str = "x",
        y_col: str = "y",
        r_col: str = "R",
        g_col: str = "G",
        b_col: str = "B",
        region_col: str = "unique_region",
    ) -> pd.DataFrame:
        """
        Add RGB columns to a dataframe and rasterize each region to disk.

        This method first encodes the model input columns into RGB triplets and
        appends them to the dataframe, then rasterizes the RGB outputs separately
        for each region.

        Args:
            result_df: Input dataframe containing all required autoencoder input
                columns, coordinate columns, and a region identifier column.
            save_dir: Directory where rasterized region outputs will be written.
            split: Whether to split regions into train/val subdirectories.
                Set to False for inference, where all regions go into save_dir.
            val_ratio: Fraction of regions assigned to the validation split (ignored when split=False).
            image_size: Side length, in pixels, of each rasterized output image.
            file_name_prefix: Prefix used when naming saved rasterized files.
            seed: Random seed used for deterministic train/validation splitting.
            x_col: Name of the dataframe column containing x coordinates.
            y_col: Name of the dataframe column containing y coordinates.
            r_col: Name of the red RGB output column to add.
            g_col: Name of the green RGB output column to add.
            b_col: Name of the blue RGB output column to add.
            region_col: Name of the dataframe column identifying regions.

        Returns:
            A copy of ``result_df`` with added RGB columns.

        Raises:
            ValueError: If required rasterization columns are missing, if RGB/output
                column names collide with rasterization columns, or if scalar
                rasterization parameters are invalid.
        """
        save_dir = Path(save_dir)

        self._validate_required_dataframe_columns(
            result_df,
            columns=[x_col, y_col, region_col],
            context="rasterization",
        )
        self._validate_distinct_column_names(
            columns=[x_col, y_col, region_col, r_col, g_col, b_col],
            names=["x_col", "y_col", "region_col", "r_col", "g_col", "b_col"],
            context="Rasterization/input-output",
        )
        self._validate_rasterization_params(
            val_ratio=val_ratio,
            image_size=image_size,
            file_name_prefix=file_name_prefix,
        )

        result_df = self._encode_to_rgb_df(
            df=result_df,
            r_col=r_col,
            g_col=g_col,
            b_col=b_col,
        )

        rasterize_rgb_regions(
            result_df=result_df,
            save_dir=save_dir,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            region_col=region_col,
            x_col=x_col,
            y_col=y_col,
            r_col=r_col,
            g_col=g_col,
            b_col=b_col,
            image_size=image_size,
            filename_prefix=file_name_prefix,
        )

        return result_df