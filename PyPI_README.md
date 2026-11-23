# MORPHE

Spatial omics embedding and cascaded diffusion for inpainting, outpainting, 2D imputation and 3D imputation.

**Author:** yuan feng ([yf185@duke.edu](mailto:yf185@duke.edu)). Distribution: `morphe-spatial`; Python import and CLI: `morphe`.

Based on [HickeyLab/MORPHE](https://github.com/HickeyLab/MORPHE). Source provenance is in `morphe-examples/UPSTREAM.md`; the workflow guide is in `morphe-examples/docs/USAGE.md` after exporting the documentation below.

## Quick start: open the notebooks after pip installation

Use the same Python environment where you installed `morphe-spatial`. If you use
the reference Conda environment, activate it with `conda activate morphe` first.
The `notebooks` extra installs all workflow dependencies and the JupyterLab,
ipykernel, nbformat and matplotlib tools used by the eight default notebooks.
After installation, run `morphe` or `python -m morphe` to display the quick start;
run `python -m morphe readme` to display the complete README in the terminal.
Standard pip wheel installation does not automatically display or open a README.

```bash
# 1. Install the workflow and notebook tools in the current Python environment.
python -m pip install 'morphe-spatial[notebooks]'

# 2. Register that Python as the notebook kernel.
python -m ipykernel install --user --name morphe --display-name 'Python (MORPHE cu121)'

# 3. Export the notebooks, saved visualizations and documentation.
python -m morphe examples --output ./morphe-examples

# 4. Open the exported notebook directory in JupyterLab.
python -m jupyterlab --notebook-dir=./morphe-examples/notebooks
```

`--output ./morphe-examples` creates a folder named `morphe-examples` in the
current terminal directory. The notebooks are inside `morphe-examples/notebooks/`;
they are copied out of the installed package so you can edit and run them.
JupyterLab opens this directory directly. If a browser does not open automatically,
open the local URL printed in the terminal. If JupyterLab is already running,
export the files from the same working directory and browse to this folder in
the existing session instead of launching another process.

These commands use your existing PyTorch installation; check its GPU support with
`python -m morphe doctor` before training. Use the fresh-environment setup if you
need a new CUDA environment. For an older or edited export, choose
a new `--output` folder and use the same folder in the JupyterLab command.

The complete **train and infer notebooks for every task are in the examples folder**. Open the matching notebook below and choose **Python (MORPHE cu121)**. Each stage explains its inputs, outputs, model files and intermediate visualizations.

| Task | Complete training example | Complete inference example |
| --- | --- | --- |
| Inpainting | `morphe-examples/notebooks/01_inpainting_train.ipynb` | `morphe-examples/notebooks/02_inpainting_infer.ipynb` |
| Outpainting | `morphe-examples/notebooks/03_outpainting_train.ipynb` | `morphe-examples/notebooks/04_outpainting_infer.ipynb` |
| 2D imputation | `morphe-examples/notebooks/05_imputation2d_train.ipynb` | `morphe-examples/notebooks/06_imputation2d_infer.ipynb` |
| 3D imputation | `morphe-examples/notebooks/07_imputation3d_train.ipynb` | `morphe-examples/notebooks/08_imputation3d_infer.ipynb` |

In JupyterLab:

1. Double-click the matching `*_train.ipynb` or `*_infer.ipynb` file.
2. Select **Kernel -> Change Kernel -> Python (MORPHE cu121)**.
3. Review the saved visualizations. In the second code cell, set the input data path and `TRAIN_DIR` for training, or the trained project `LOAD_DIR` for inference.
4. Run one cell at a time with **Shift+Enter**. Use a separate kernel for inference. Restart kernels without clearing the included sample outputs.

## Fresh Linux CUDA environment

Use this route only when creating a new environment. The quick start above is
for users who already installed MORPHE. These Linux x86_64 instructions require
Conda and a working NVIDIA driver. The reference environment is Python 3.12,
PyTorch 2.5.1 and CUDA 12.1. Choose a new working directory for this setup.

```bash
# 1. Create and activate the Python environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n morphe python=3.12 pip -y
conda activate morphe
python -m pip install --upgrade pip

# 2. Create a workspace and write its reference constraints file.
mkdir -p morphe-work
cd morphe-work
cat > constraints-cu121.txt <<'EOF'
# Linux / Python 3.12 / RTX A6000 reference target. Install CUDA torch first.
torch==2.5.1
torchvision==0.20.1
diffusers==0.32.2
transformers==4.46.3
accelerate==1.12.0
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
torch-geometric==2.6.1
EOF

# 3. Install the CUDA PyTorch pair.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 4. Install MORPHE, workflow dependencies and notebook tools.
python -m pip install -c constraints-cu121.txt 'morphe-spatial[notebooks]'
python -m pip check

# 5. Check package imports and CUDA.
python -c "import morphe,torch; from morphe.notebook import run_cli; print(morphe.__version__,morphe.__file__); print(torch.__version__,torch.cuda.is_available()); assert torch.cuda.is_available()"
python -m morphe doctor

# 6. Register the notebook kernel.
python -m ipykernel install --user --name morphe --display-name 'Python (MORPHE cu121)'

# 7. Choose a writable model cache and GPU.
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HOME/.cache/morphe/huggingface"
mkdir -p "$HF_HOME"

# 8. Export the bundled examples and open their notebook directory.
python -m morphe examples --output ./morphe-examples
python -m jupyterlab --notebook-dir=./morphe-examples/notebooks
```

After step 8, open the matching train or infer notebook listed above and select **Python (MORPHE cu121)**. All eight complete examples are in `morphe-examples/notebooks/`.

For an installation from a downloaded wheel, replace the MORPHE installation in step 4 with `python -m pip install -c constraints-cu121.txt "/path/to/morphe_spatial-0.2.16-py3-none-any.whl[notebooks]"`. The remaining steps are the same.

## Examples included with the package

Both the wheel and source distribution contain the eight notebooks and their saved visual outputs. The README is also displayed on the PyPI project page. The export command in the quick start creates `morphe-examples/notebooks/`, `morphe-examples/docs/`, the README/environment guide and the CUDA constraints file. Existing files with different content are preserved; choose another output directory to export a fresh copy. The source archive and release ZIP already contain the examples folder. The table lists local file paths. PyPI/TestPyPI displays this README but does not host these files as web pages. Open the exported files in JupyterLab or a Markdown viewer. Use a new output directory when upgrading an existing exported copy.

## Inputs and saved models

The three 2D workflows accept CSV (`x`, `y`, `unique_region`, `Cell Type`, biomarkers) or H5AD. The full 3D workflow accepts separate mouse1 training and mouse2 test H5AD files. Coordinate, section, label and marker mappings are configurable in each training notebook. Replace `/path/to/data/...` with your data paths. Relative `./runs/...` paths use the kernel working directory.

Each task saves its own classifier, autoencoder, RGB min/max, latent diffusion and pixel diffusion components. Inference explicitly selects `LOAD_DIR`, restores the matching AE min/max and exports a pixel-level CSV cell map using the original class names. The 2D default inputs come from the saved test-region split; explicit RGB image paths take priority.

Published initialization weights are resolved by task from Hugging Face and cached separately. They are not bundled in the wheel. The task notebooks show every training and inference stage separately.

## Documentation

After the quick-start export, open these local files in JupyterLab or a Markdown viewer:

| Guide | Exported file |
| --- | --- |
| Examples and task settings | `morphe-examples/notebooks/README.md` |
| Environment guide | `morphe-examples/notebooks/ENVIRONMENT.md` |
| CSV and H5AD field mappings | `morphe-examples/docs/INPUT_FORMATS.md` |
| 3D H5AD workflow | `morphe-examples/docs/H5AD_3D.md` |
| API and data contracts | `morphe-examples/docs/API.md` |
