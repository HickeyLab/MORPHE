import os, numpy as np, matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torchvision import utils as vutils

from data.dataset_cascade import PrecomputedCascadeDataset
from models.latent_adapter import LatentAdapter
from models.unet512 import UNet512
from utils.composition import infer_cell_map, compute_type_distribution

from diffusers import DDPMScheduler

class Cascade512Trainer:
    ...
    （你的 Trainer 全部放在这里）
