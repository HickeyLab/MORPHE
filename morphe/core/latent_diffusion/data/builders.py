from pathlib import Path

from .datasets import InpaintDataset, OutpaintDataset, Slice3DDataset


def build_inpaint_dataset(
    input_dir: Path,
    masks_per_image_train: int = 2,
    masks_per_image_val: int = 5,
    img_size: int = 512,
) -> tuple[InpaintDataset, InpaintDataset]:
    """
    Build the train and validation datasets for the inpainting task.

    This expects ``input_dir`` to contain ``train`` and ``val`` subdirectories.
    Separate mask counts can be configured for training and validation so that
    data generation behavior can differ between the two splits.

    Args:
        input_dir: Root directory containing the dataset splits.
        masks_per_image_train: Number of masks to generate per training image.
        masks_per_image_val: Number of masks to generate per validation image.
        img_size: Image size used when loading and preprocessing samples.

    Returns:
        A tuple of ``(train_dataset, val_dataset)`` as ``InpaintDataset``
        instances.
    """
    train_dir = input_dir / "train"
    val_dir = input_dir / "val"

    train_dataset = InpaintDataset(
        train_dir,
        img_size=img_size,
        masks_per_image=masks_per_image_train,
    )

    val_dataset = InpaintDataset(
        val_dir,
        img_size=img_size,
        masks_per_image=masks_per_image_val,
    )

    return train_dataset, val_dataset


def build_outpaint_dataset(
    input_dir: Path,
    img_size: int = 512,
    masks_per_image_train: int = 5,
    masks_per_image_val: int = 5,
) -> tuple[OutpaintDataset, OutpaintDataset]:
    """
    Build the train and validation datasets for the outpainting task.

    This expects ``input_dir`` to contain ``train`` and ``val`` subdirectories.
    A separate mask count can be provided for each split.

    Args:
        input_dir: Root directory containing the dataset splits.
        img_size: Image size used when loading and preprocessing samples.
        masks_per_image_train: Number of masks to generate per training image.
        masks_per_image_val: Number of masks to generate per validation image.

    Returns:
        A tuple of ``(train_dataset, val_dataset)`` as ``OutpaintDataset``
        instances.

    Raises:
        ValueError: If ``input_dir`` is not provided.
    """
    if not input_dir:
        raise ValueError("No input_dir provided.")

    train_dir = input_dir / "train"
    val_dir = input_dir / "val"

    return (
        OutpaintDataset(train_dir, img_size, masks_per_image_train),
        OutpaintDataset(val_dir, img_size, masks_per_image_val),
    )


def build_three_dim_dataset(
    input_dir: Path,
) -> tuple[Slice3DDataset, Slice3DDataset]:
    """
    Build the datasets for the 3D slice task.

    The current implementation returns the same dataset construction for both
    train and validation outputs.

    Args:
        input_dir: Root directory containing the 3D slice data.

    Returns:
        A tuple of ``(train_dataset, val_dataset)`` as ``Slice3DDataset``
        instances.

    Raises:
        ValueError: If ``input_dir`` is not provided.
    """
    if not input_dir:
        raise ValueError("No input_dir provided.")

    return (Slice3DDataset(input_dir), Slice3DDataset(input_dir))