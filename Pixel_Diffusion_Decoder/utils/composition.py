import torch
import numpy as np

def infer_cell_map(img, ae_model):
    """
    Runs autoencoder → returns type map
    """
    with torch.no_grad():
        return ae_model(img.unsqueeze(0))

def compute_type_distribution(type_map, num_types=25):
    """
    Compute histogram of cell types
    """
    flat = type_map.flatten()
    hist, _ = np.histogram(flat, bins=num_types, range=(0, num_types))
    return hist / (hist.sum() + 1e-8)
