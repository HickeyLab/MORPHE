# DISCO
DISCO


DISCO (Diffusion for Single-Cell-Level Organization), a generative framework that learns directly from spatial-omics data to synthesize biologically faithful tissue architecture.

File Structure

--Main
	|--Embeddings: Jupyter notebooks used for the GCNN and Autoencoder. Also includes STELLAR application to our data.
	|--Evaluation: Notebooks for novel model and output evaluation metrics and comparisons.
	|	|--Neighborhoods: Spatial Omics neighborhood analysis evaluation.
	|--GapFiller: Gap filler diffusion and inference. Also evaluation for GapFiller by cell type composition.
	|--Inpainting: Inpainting fine-tuning, bootstrap inpainting.
	|--Merfish: DISCO applies to MERFISH spatial transcriptomics data.
	|--Outpainting: Outpainting notebook and python script
	|--Preprocess: **Need to update with notebooks**
	|--Recoveries: Notebooks used to test color recovery, and also other types of diffusion.
	|--train_and_inference: Cascaded stable diffusion training, decoding inference using diffusion decoder.
	|--Outputs: **Need to update with outputs/data files from google collab**
