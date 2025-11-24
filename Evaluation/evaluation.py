import torch
from metrics import (cluster_cell_types, neighborhood_similarity,
                     cell_type_distribution,
                     spatial_distribution_fidelity, calculate_comprehensive_score)
import numpy as np
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, in_dim=25, bottleneck_dim=3, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/4), int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/4), bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, int(hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/4), int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/4), in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

def infer_cell_map(latent_image, model):
    """
    latent_image: torch.Tensor of shape [3, H, W], normalized to [0, 1] or [-1, 1]
    model: trained Autoencoder (only decoder is used)

    Returns:
        pred_image: torch.Tensor of shape [1, H, W], with predicted cell type index (0~24),
                    pure white points are directly assigned as 0
    """
    H, W = latent_image.shape[1], latent_image.shape[2]
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        # Flatten image: [3, H, W] → [H*W, 3]
        flat_img = latent_image.permute(1, 2, 0).reshape(-1, 3).to(device)

        # Mask: pure white pixel (assume normalized to [0, 1])
        white_mask = (flat_img == 1.0).all(dim=1)  # [H*W], bool

        # Prepare input for model only where not pure white
        infer_input = flat_img[~white_mask]  # [N_nonwhite, 3]
        pred = torch.zeros(flat_img.shape[0], dtype=torch.long, device=device)  # [H*W]

        if infer_input.shape[0] > 0:
            logits = model.decoder(infer_input)  # [N_nonwhite, 25]
            pred_nonwhite = torch.argmax(logits, dim=1)  # [N_nonwhite]
            pred[~white_mask] = pred_nonwhite  # Assign to result

        pred_image = pred.reshape(1, H, W)  # [1, H, W]

    return pred_image


from PIL import Image
from torchvision import transforms


import torch
from PIL import Image


def load_and_recover_z3d_png(path, min_vals=None, range_vals=None):
    """
    Load a PNG image as RGB, interpret it as normalized z_3d embedding encoded as uint8 [0–255],
    and recover the original z_3d float values via inverse scaling.

    Args:
        path: Path to PNG image.
        min_vals: numpy array or list of shape [3], original encoder min per dim.
        range_vals: numpy array or list of shape [3], original encoder range per dim.

    Returns:
        torch.Tensor of shape [3, H, W], float32, recovered latent values.
    """
    # 1. Load RGB image as uint8 tensor
    img = Image.open(path).convert("RGB").resize((512, 512), resample=Image.NEAREST)
    arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    rgb = arr.view(img.size[1], img.size[0], 3).permute(2, 0, 1).clone()  # [3, H, W], uint8

    # 2. Convert to float in [0,1]
    rgb_float = rgb.float() / 255.0  # [3, H, W]
    '''

    # 3. Recover latent z_3d: z = rgb * range + min
    # min_vals and range_vals are [3] arrays (numpy or list), we convert to tensor
    min_tensor = torch.tensor(min_vals, dtype=torch.float32).view(3, 1, 1)
    range_tensor = torch.tensor(range_vals, dtype=torch.float32).view(3, 1, 1)

    z_3d = rgb_float * range_tensor + min_tensor  # [3, H, W], float32
    '''

    return rgb_float


import torch

def rgb_images_to_shared_color_classes(img1, img2):
    """
    Convert two [3, H, W] RGB float images in [0,1] range into consistent label maps.
    White (255,255,255) → label 0; other unique RGBs → 1,2,...

    Parameters:
        img1, img2: [3, H, W], float32, range [0, 1]

    Returns:
        label_map1, label_map2: [1, H, W], torch.long
    """
    # 1. Convert float RGB [0,1] → uint8 [0,255]
    img1_uint8 = (img1 * 255).round().clamp(0, 255).to(torch.uint8)
    img2_uint8 = (img2 * 255).round().clamp(0, 255).to(torch.uint8)

    # 2. Flatten to [N, 3]
    flat1 = img1_uint8.permute(1, 2, 0).reshape(-1, 3)
    flat2 = img2_uint8.permute(1, 2, 0).reshape(-1, 3)
    all_colors = torch.cat([flat1, flat2], dim=0)  # [N1+N2, 3]

    # 3. Unique color rows → inverse gives per-pixel label
    unique_colors, inverse = torch.unique(all_colors, dim=0, return_inverse=True)

    # 4. Force white [255,255,255] to label 0
    white = torch.tensor([255, 255, 255], dtype=torch.uint8, device=img1.device)
    white_mask = (unique_colors == white).all(dim=1)

    if white_mask.any():
        white_idx = white_mask.nonzero(as_tuple=True)[0].item()
        if white_idx != 0:
            # swap white color to index 0
            unique_colors[[0, white_idx]] = unique_colors[[white_idx, 0]]
            # adjust label map
            label_map = inverse.clone()
            label_map[inverse == 0] = white_idx
            label_map[inverse == white_idx] = 0
        else:
            label_map = inverse
    else:
        label_map = inverse

    # 5. Reshape to [1, H, W]
    H, W = img1.shape[1:]
    label_map1 = label_map[:H*W].reshape(1, H, W)
    label_map2 = label_map[H*W:].reshape(1, H, W)

    return label_map1.long(), label_map2.long()





if __name__ == '__main__':
    true_map = load_and_recover_z3d_png("C:/Users/18983/Desktop/1.png")
    generated_map = load_and_recover_z3d_png("C:/Users/18983/Desktop/0.png")
    model = Autoencoder().to("cuda")
    model.load_state_dict(torch.load('autoencoder_weights.pth'))
    true_map = infer_cell_map(true_map, model)
    generated_map = infer_cell_map(generated_map, model)
    #true_map, generated_map = rgb_images_to_shared_color_classes(true_map, generated_map)
    print(len(torch.unique(true_map)))
    print("True map shape:", true_map.shape)
    print("Generated map shape:", generated_map.shape)

    #   Evaluate each individual metric
    clustering_results = cluster_cell_types(true_map, generated_map, num_clusters=25)
    cell_type_clustering_score = clustering_results['ARI']  # You can choose ARI or NMI
    neighborhood_similarity_score = neighborhood_similarity(true_map, generated_map)
    cell_type_distribution_score = cell_type_distribution(true_map, generated_map)
    spatial_distribution_fidelity_score = spatial_distribution_fidelity(true_map, generated_map)

    print("clustering_results", clustering_results)
    print("cell_type_clustering_score", cell_type_clustering_score)
    print("neighborhood_similarity_score", neighborhood_similarity_score)
    print("cell_type_distribution_score", cell_type_distribution_score)
    print("spatial_distribution_fidelity_score", spatial_distribution_fidelity_score)


    # Compute the comprehensive score by combining all the metrics
    comprehensive_score = calculate_comprehensive_score(
        cell_type_clustering_score,
        neighborhood_similarity_score,
        cell_type_distribution_score,
        spatial_distribution_fidelity_score
    )

    # Print the comprehensive score
    print("Comprehensive Score:", comprehensive_score)