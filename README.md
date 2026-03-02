# MORPHOS

## Overview
*MORPHOS: Bridging Image Generation and Spatial Omics
for Tissue Synthesis*

**Authors:**  
Yuan Feng¹, Zachary Robers², Leyla Rasheed², Yang Miao¹, Shuo Wen³,  
James Sohigian², Maria Brbić³, John W. Hickey¹\*

¹ Department of Biomedical Engineering, Duke University, Durham, NC 27708, USA  
² Department of Computer Science, Duke University, Durham, NC 27708, USA  
³ School of Computer and Communication Sciences, EPFL, Lausanne CH-1015, Switzerland  


**Preprint:**  
[Link to bioRxiv]

### MORPHOS Pipeline

<p align="center">
  <img src="Assets/Fig1.png" width="800">
</p>

### Abstract
Spatially resolved omics technologies reveal tissue organization at single-cell resolution but remain
limited by the cost of the assays, incomplete spatial coverage, 2D-only imaging, and experi-
mental artifacts. These factors motivate the need for in silico methods that can reconstruct or
extend tissue context beyond what current spatial measurements provide. We present MOR-
PHOS (Modeling Organized Representations of Probabilistic Hierarchical Organization in Space),
an AI framework that learns to synthesize biologically faithful tissue architecture directly from
spatial-omics data. MORPHOS introduces a graph-informed probabilistic embedding that maps
discrete cell identities and their spatial relationships into a continuous RGB-like latent space
compatible with diffusion modeling. This representational bridge enables spatial cellular maps
to leverage large pre-trained image-generative models while preserving biological interpretabil-
ity upon decoding. By modeling cells as the fundamental units of generation and learning how
their identities and spatial relationships collectively give rise to large-scale tissue structure, MOR-
PHOS enables generation and reconstruction of tissue architecture at single-cell resolution. We
applied the method across large-scale single-cell proteomic datasets from the intestine and single-
cell transcriptomic datasets from the brain, showing computational scalability acrosss millions of
cells. We used MORPHOS on these datasets to outpaint beyond experimentally restricted fields
of view, inpaint missing or experimentally damaged tissue regions, and perform cross-tissue impu-
tation, connecting separated tissue regions into a single contiguous sample in both 2D and 3D.
MORPHOS represents a new class of tissue generation algorithms that will help solve current
limitations and challenges with single-cell spatial-omics datasets.




## Repository Structure (Python Notebooks)
```text
DISCO/
│
├── Embeddings/                         # GCNN & Autoencoder
│   ├── 01_GCN_classifier.ipynb
│   ├── 02_Autoencoder.ipynb
│   └── 03_Interpret_cellmap.ipynb  
├── Evaluation/                         # Evaluation Metrics
├── Finetune/                           # Latent Diffusion Finetune 
├── Latent_Diffusion_Generator/         # Application Train/Infer (In/Outpainting, 3D, gapfill)
├── Pixel_Diffusion_Decoder/            # Pixel Diffusion Train and Decode
├── Preprocessing/                      # DISCO Preprocessing Pipeline
│   ├── 01_Conflict_statistics_and_cleaning.ipynb
│   ├── 02_Resolution_Reduction.ipynb
│   └── 03_Embedding_DATAQuality_minimum.ipynb
├── src/disco/                          # 
└── README.md
```
## Recommended execution order

### Preprocessing Pipeline

1. `Preprocessing/01_Conflict_statistics_and_cleaning.ipynb`
2. `Preprocessing/02_Resolution_Reduction.ipynb`
3. `Preprocessing/03_Embedding_DATAQuality_minimum.ipynb` 

### Embedding pipeline
1. `Embeddings/01_GCN_classifier.ipynb` — learn neighborhood-informed representations  
2. `Embeddings/02_Autoencoder.ipynb` — compress representations into latent space  
3. `Embeddings/03_Interpret_cellmap.ipynb` — decode end-to-end outputs back to original cell identities



### Baselines
- `baselines/MLP_classifier.ipynb` — legacy baseline classifier (not part of the final DISCO pipeline)
