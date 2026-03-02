# MORPHE

## Overview
*MORPHE: Bridging Image Generation and Spatial Omics
for Tissue Synthesis*

**Authors:**  
Yuan Feng¹, Zachary Robers², Leyla Rasheed², Yang Miao¹, Shuo Wen³,  
James Sohigian², Maria Brbić³, John W. Hickey¹\*

¹ Department of Biomedical Engineering, Duke University, Durham, NC 27708, USA  
² Department of Computer Science, Duke University, Durham, NC 27708, USA  
³ School of Computer and Communication Sciences, EPFL, Lausanne CH-1015, Switzerland  


**Preprint:**  
[Link to bioRxiv]

### MORPHE Pipeline

<p align="center">
  <img src="Assets/Fig 1.png" width="800">
</p>

### Abstract
Spatially resolved omics technologies reveal tissue organization at single-cell resolution but remain
limited by the cost of the assays, incomplete spatial coverage, 2D-only imaging, and experi-
mental artifacts. These factors motivate the need for in silico methods that can reconstruct or
extend tissue context beyond what current spatial measurements provide. We present MORPHE (MOdeling of stRuctured sPatial High-dimensional Embeddings), an AI framework that learns to synthesize biologically faithful tissue architecture directly from
spatial-omics data. MORPHE introduces a graph-informed probabilistic embedding that maps
discrete cell identities and their spatial relationships into a continuous RGB-like latent space
compatible with diffusion modeling. This representational bridge enables spatial cellular maps
to leverage large pre-trained image-generative models while preserving biological interpretabil-
ity upon decoding. By modeling cells as the fundamental units of generation and learning how
their identities and spatial relationships collectively give rise to large-scale tissue structure, MORPHE enables generation and reconstruction of tissue architecture at single-cell resolution. We
applied the method across large-scale single-cell proteomic datasets from the intestine and single-
cell transcriptomic datasets from the brain, showing computational scalability acrosss millions of
cells. We used MORPHE on these datasets to outpaint beyond experimentally restricted fields
of view, inpaint missing or experimentally damaged tissue regions, and perform cross-tissue impu-
tation, connecting separated tissue regions into a single contiguous sample in both 2D and 3D.
MORPHE represents a new class of tissue generation algorithms that will help solve current
limitations and challenges with single-cell spatial-omics datasets.

### MORPHE Usecases
https://github.com/user-attachments/assets/f80406a2-3ffc-4cbd-93fd-7968fe4b1885


## Repository Structure (Python Notebooks)
```text
DISCO/
│
├── Assets/                         
├── Alternatives/
│   └── MLP_classifier.ipynb                   # MLP Classifier, an alternative method for GCN Classifier
├── Embeddings/                             # GCNN Classifier & Autoencoder
│   ├── 01_GCN_classifier.ipynb
│   ├── 02_Autoencoder.ipynb
│   └── 03_Interpret_cellmap.ipynb  
├── Evaluation/                             # Evaluation Metrics
│   └── Evaluation.ipynb  
├── Finetune/                               # Comparisons with Existing Image-Generative Models
│   ├── Fluxfill/
│   │   ├── Train_Fluxfill.py
│   │   └── Fluxfill_outpainting.py
│   └── SD2/ 
│       ├── Train_SD2.py
│       └── SD2_Outpainting.py
├── Latent_Diffusion_Generator/             # Code of training & inference for all 4 usecases
│   ├── 3D_Imputation/
│   │   ├── Train_3D_Imputation.ipynb
│   │   └── Infer_3D_Imputation.ipynb
│   ├── Arbitrary_Inpainting/
│   │   ├── Train_Arbitrary_Inpainting.ipynb
│   │   └── Infer_Arbitrary_Inpainting.ipynb
│   └── Outpainting_and_2D_Imputation/
│       ├── Train/                          # Library-Style Code Folder for training the model
│       └── Inferences/
│           ├── Inference_2D_Imputation.ipynb
│           └── Inference_Outpainting.ipynb
├── Pixel_Diffusion_Decoder/                # Pixel Diffusion, used to refine the output of Latent_Diffusion_Generator (whatever the usecase is)
├── Preprocessing/                          # Preprocessing Pipeline
│   └── Resolution_Reduction.ipynb
│                      
└── README.md
```
## Recommended execution order

### 1. Preprocessing Pipeline (Optional)

Input: Segmented and annotated spatial omics cell maps (Columns must include x,y coordinate, cell type, molecular expression vector).

1. `Preprocessing/resolution_reduction.ipynb` - Reduce resolution using dimensions and conflict cleaned data.
   
Output: Resolution reduced spatial omics cell maps.

### 2. Embedding Pipeline

Input: Processed spatial omics cell maps.

1. `Embeddings/01_GCN_classifier.ipynb` — Output spatial and marker informed probabilities.
2. `Embeddings/02_Autoencoder.ipynb` — compress probabilities to three-channel representation of pixel intensities

Output: Embedded image for each cell map.

### 3. Latent Diffusion Generator. Choose one of the usecases for training & inference here: 1. Arbitrary Inpainting 2. 2D Imputation & Outpainting (share one training folder) 3. 3D Imputation.

Input: Split images (.pngs) into training and val set.

#### Arbitrary Inpainting
1. `Latent_Diffusion_Generator/Arbitrary_Inpainting/Train_Arbitrary_Inpainting.ipynb`
2. `Latent_Diffusion_Generator/Arbitrary_Inpainting/Infer_Arbitrary_Inpainting.ipynb`

#### 2D Imputation & Outpainting
1. `Latent_Diffusion_Generator/Outpainting_and_2D_Imputation/Train', use this library-style folder for training
2. Inference, choose one of them based on your task:
- `Latent_Diffusion_Generator/Outpainting_and_2D_Imputation/Inference/Inference_2D_Imputation.ipynb`
- `Latent_Diffusion_Generator/Outpainting_and_2D_Imputation/Inference/Inference_Outpainting.ipynb`

#### 3D Imputation
1. `Latent_Diffusion_Generator/3D_Imputation/Train_3D_Imputation.ipynb`
2. `Latent_Diffusion_Generator/3D_Imputation/Infer_3D_Imputation.ipynb`

Output: Raw Latent feature (".pt" files, shape: 4×64×64) for the refining module.

### 4. Refining. Refine the output from Latent Diffusion Generator (whichever the usecase is).

Input: Raw Latent feature (".pt" files, shape: 4×64×64)

1. `Pixel_Diffusion_Decoder/Dataset_Construct_for_Decoder.ipynb` for constructing the dataset for the refining diffusion model. (Note that you need put some ".png" files (shape: 3×512×512) in the "root_dir" for pixel_diffusion to learn reconstructing high-resolution images from latent feature)
2. `Pixel_Diffusion_Decoder/train_decoder.py` for training.
3. `Pixel_Diffusion_Decoder/Infer_Decoder.ipynb` for inference.

Ouput: High-Resolution Images (".png" files, shape: 3×512×512)

### 5. Decoding

Input: High-Resolution Images (".png" files, shape: 3×512×512)

1. `Embeddings/03_Interpret_cellmap.ipynb` — decode from images back into original cell identities creating a new spatial omics cell map.
   
Output: 1×512×512 '.pt' files, each pixel is annotated with a type of cell.
 
### 6. Evaluation
`Evaluation/Evaluation.ipynb`

- RGB Centroid Distance Score  
- Neighbor KMeans Composition Matching Score  
- Cell Density Score  
- Spatial Structure Score  
- Cell Type Distribution Score  

#### Optional 
Alternative

- `Alternative/MLP_classifier.ipynb` - baseline classifier for comparison to GCNN

Fine-tune Existign Image Generative Models (FluxFill and StableDiffusion2):

- `Finetune/Fluxfill/Train_Fluxfill.py`
- `Finetune/SD2/Train_SD2.py`

#### Affiliations
<p align="left">
  <img src="Assets/logo.png" width="110" style="margin-right:-10px;">
  <img src="Assets/logo2.png" width="175">
</p>
