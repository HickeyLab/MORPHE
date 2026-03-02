from pathlib import Path
import torch

from src.disco.core.latent_diffusion.diffusion_trainer import DiffusionTrainer
from src.disco.core.latent_diffusion.artifact import LatentDiffuserArtifact
from src.disco.core.latent_diffusion.strategy.outpaint import OutpaintDiffusion
from src.disco.core.latent_diffusion.infer.outpainting import OutpaintInferencer


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

DATA_ROOT = Path("outpaint_images")   # 必须包含 train/ 和 val/
TEST_IMAGE = Path("infer_test_image.png")

CHECKPOINT_DIR = Path("checkpoints")
EPOCHS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------
# TRAIN
# -------------------------------------------------------

def train():

    strategy = OutpaintDiffusion(
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
        bbox_encoder_kwargs = {"in_dim": 4, "out_dim": 32,},
        batch_size=1,
        val_batch_size=1,
    )

    trainer.train(
        epochs=EPOCHS,
        show_loss_curve=True,
    )

    print("Outpaint training finished.")


# -------------------------------------------------------
# INFERENCE
# -------------------------------------------------------

def inference():

    artifact_path = CHECKPOINT_DIR / "latent_diffuser_artifact.pt"

    if not artifact_path.exists():
        raise RuntimeError("Artifact not found. Train first.")

    artifact = LatentDiffuserArtifact.load(artifact_path)

    strategy = OutpaintDiffusion(
        img_size=512,
        train_num_workers=0,
        val_num_workers=0,
    )

    inferer = OutpaintInferencer(
        artifact=artifact,
        strategy=strategy,
        device=DEVICE,
    )

    result = inferer.run_one(
        original_file_path=TEST_IMAGE,
        save_dir="outpaint_results",
        save_name="demo",
        steps=200,
        crop_ratio=0.97,
        iterations=5,
        direction="right",
        show_plot=True,
        plot_title="Outpaint Result",
    )

    print("Outpaint inference complete.")
    print("Saved directory:", result.extras["run_dir"])


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":

    print("==== OUTPAINT TRAINING START ====")
    #train()

    print("==== OUTPAINT INFERENCE START ====")
    inference()