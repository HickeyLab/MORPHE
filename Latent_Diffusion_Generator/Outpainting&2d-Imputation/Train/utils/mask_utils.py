import torch

def create_latent_mask(bbox, latent_shape, device):
    b, _, H, W = latent_shape
    masks = []

    for coords in bbox:
        x1, y1, x2, y2 = coords * torch.tensor([W, H, W, H], device=device)
        xx, yy = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device)
        )
        mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()
        masks.append(mask)

    return torch.stack(masks).unsqueeze(1)
