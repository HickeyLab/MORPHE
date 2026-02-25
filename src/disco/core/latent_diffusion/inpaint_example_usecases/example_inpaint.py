import torch
from pathlib import Path

from src.disco.core.latent_diffusion.diffusion_trainer import DiffusionTrainer
from src.disco.core.latent_diffusion.artifact import LatentDiffuserArtifact
from src.disco.core.latent_diffusion.strategy.arbitrary_inpainting import ArbitraryInpainting
from src.disco.core.latent_diffusion.infer.arbitrary_inpainting import ArbitraryInpaintingInferer


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATA_ROOT = Path("inpaint_images")
TEST_IMAGE = Path("infer_test_image.png")
TEST_MASK = Path("infer_test_mask.png")

CHECKPOINT_DIR = Path("checkpoints")
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------
# TRAIN
# -------------------------------------------------------

def train():

    strategy = ArbitraryInpainting(
        img_size=512,
        train_num_workers=0,
        val_num_workers=0,
        patience=1,
        decay_enabled=False,
    )

    trainer = DiffusionTrainer(
        strategy=strategy,
        root_dir=DATA_ROOT,
        save_dir=CHECKPOINT_DIR,
        pretrained="runwayml/stable-diffusion-v1-5",
        lr=2e-5,
        mixed_precision="fp16",
        grad_clip=1.0,
        cond_encoder_kwargs={"num_tokens": 64},
        coord_encoder_kwargs={"embed_dim": 32, "num_tokens": 64},
        batch_size=1,
        val_batch_size=1,
    )

    trainer.train(
        epochs=EPOCHS,
        show_loss_curve=True,
    )

    print("Training finished.")


# -------------------------------------------------------
# INFERENCE
# -------------------------------------------------------

def inference():

    artifact_path = CHECKPOINT_DIR / "latent_diffuser_artifact.pt"

    if not artifact_path.exists():
        raise RuntimeError("Artifact not found. Train first.")

    artifact = LatentDiffuserArtifact.load(artifact_path)

    strategy = ArbitraryInpainting(
        img_size=512,
        train_num_workers=0,
        val_num_workers=0,
    )

    inferer = ArbitraryInpaintingInferer(
        artifact=artifact,
        strategy=strategy,
        device=DEVICE,
    )

    result = inferer.run_one(
        image_path=TEST_IMAGE,
        mask_path=TEST_MASK,
        num_steps=200,
        show_plot=True,
        plot_title="Inpainting Result",
    )

    print("Inference complete.")
    print("Extra info keys:", result.extras.keys())


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":

    print("==== TRAINING START ====")
    #train()

    print("==== INFERENCE START ====")
    inference()