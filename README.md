# DISCO
DISCO


DISCO (Diffusion for Single-Cell-Level Organization), a generative framework that learns directly from spatial-omics data to synthesize biologically faithful tissue architecture.

## File Structure

- **Main**
  - **Embeddings:** Jupyter notebooks for GCNN and autoencoder; includes STELLAR application.
  - **Evaluation:** Notebooks for model evaluation and output comparisons.
    - Neighborhoods: Spatial omics neighborhood analysis.
  - **GapFiller:** Gap-filler diffusion + inference; evaluation by cell-type composition.
  - **Inpainting:** Fine-tuning + bootstrap inpainting.
  - **Merfish:** Application to MERFISH spatial transcriptomics.
  - **Outpainting:** Outpainting notebook + Python script.
  - **Preprocess:** **Need to update with notebooks**
  - **Recoveries:** Color recovery tests + other diffusion types.
  - **train_and_inference:** Cascaded diffusion training + decoding inference.
  - **Outputs:** **Need to update with Google Colab outputs/data files**
