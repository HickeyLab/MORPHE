import math
import random
import re
import numpy as np
import os
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class Slice3DDataset(Dataset):
    """
    Filename format: <z>_<region>.png

    - prefix  z       : slice index along z-axis
    - suffix  region  : patch / region id (must stay the same)

    For each fixed region:
        choose endpoints (z_i, z_j), j >= i+2
        for every z_t where i < t < j:
            input  : (z_i, z_j)
            target : z_t
            weights: (w_prev, w_next), sum to 1
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

        # ------------------------------------------
        # Parse filenames: z_region.png
        # ------------------------------------------
        pattern = re.compile(r"^(\d+)_(\d+)\.png$")

        parsed = []  # (z, region, path)
        for fname in os.listdir(root_dir):
            m = pattern.match(fname)
            if m is None:
                continue
            z = int(m.group(1))        # prefix = z index
            region = int(m.group(2))   # suffix = region id
            parsed.append((z, region, os.path.join(root_dir, fname)))

        if len(parsed) == 0:
            raise RuntimeError("No valid <z>_<region>.png files found.")

        # ------------------------------------------
        # Group by region (suffix!)
        # ------------------------------------------
        region_groups = {}
        for z, region, path in parsed:
            region_groups.setdefault(region, []).append((z, path))

        # ------------------------------------------
        # Build interpolation samples
        # ------------------------------------------
        self.base_samples = []
        # each: (prev_path, next_path, gt_path, w_prev, w_next)

        for region, items in region_groups.items():
            items = sorted(items, key=lambda x: x[0])  # sort by z
            n = len(items)
            if n < 3:
                continue

            for i in range(n-2):
                z_i, path_i = items[i]
                for j in range(i + 2, n):
                    z_j, path_j = items[j]
                    denom = z_j - z_i
                    for t in range(i + 1, j):
                        z_t, path_t = items[t]

                        d_prev = z_t - z_i
                        d_next = z_j - z_t

                        w_prev = d_next / (d_prev + d_next)
                        w_next = d_prev / (d_prev + d_next)

                        self.base_samples.append(
                            (path_i, path_j, path_t,
                             float(w_prev), float(w_next))
                        )

        if len(self.base_samples) == 0:
            raise RuntimeError("No valid interpolation samples constructed.")

        # ------------------------------------------
        # deterministic augmentations
        # ------------------------------------------
        self.num_rotations = 4
        self.num_hflip = 2
        self.num_vflip = 2
        self.num_aug = self.num_rotations * self.num_hflip * self.num_vflip

    def __len__(self):
        return len(self.base_samples) * self.num_aug

    def __getitem__(self, idx):
        base_idx = idx // self.num_aug
        rem = idx % self.num_aug

        rot_k = rem // (self.num_hflip * self.num_vflip)
        rem %= (self.num_hflip * self.num_vflip)
        hflip = rem // self.num_vflip
        vflip = rem % self.num_vflip

        prev_path, next_path, gt_path, w_prev, w_next = \
            self.base_samples[base_idx]

        img_prev = Image.open(prev_path).convert("RGB")
        img_next = Image.open(next_path).convert("RGB")
        img_gt   = Image.open(gt_path).convert("RGB")

        if rot_k > 0:
            angle = 90 * rot_k
            img_prev = img_prev.rotate(angle)
            img_next = img_next.rotate(angle)
            img_gt   = img_gt.rotate(angle)

        if hflip:
            img_prev = img_prev.transpose(Image.FLIP_LEFT_RIGHT)
            img_next = img_next.transpose(Image.FLIP_LEFT_RIGHT)
            img_gt   = img_gt.transpose(Image.FLIP_LEFT_RIGHT)

        if vflip:
            img_prev = img_prev.transpose(Image.FLIP_TOP_BOTTOM)
            img_next = img_next.transpose(Image.FLIP_TOP_BOTTOM)
            img_gt   = img_gt.transpose(Image.FLIP_TOP_BOTTOM)

        return (
            self.to_tensor(img_prev),
            self.to_tensor(img_next),
            self.to_tensor(img_gt),
            torch.tensor(w_prev, dtype=torch.float32),
            torch.tensor(w_next, dtype=torch.float32)
        )
        
        
class OutpaintDataset(Dataset):
    def __init__(self, root_dir, img_size=512, masks_per_image=50):
        self.img_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.img_size = img_size
        self.masks_per_image = masks_per_image

        self.transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: TF.rotate(x, 90)),
                transforms.Lambda(lambda x: TF.rotate(x, 180)),
                transforms.Lambda(lambda x: TF.rotate(x, 270))
            ]),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.img_files) * self.masks_per_image

    def _gen_random_irregular_mask(self, H, W, min_area=0.01, max_area=0.1, num_vertices=10):
      cx = random.uniform(0.2, 0.8) * W
      cy = random.uniform(0.2, 0.8) * H

      target_area_ratio = random.uniform(min_area, max_area)
      target_area = target_area_ratio * H * W

      theta0 = random.uniform(0, 2*np.pi)

      angles = np.sort(np.random.uniform(0, 2*np.pi, num_vertices))

      avg_r = np.sqrt(target_area / (math.pi * 0.3))

      radii = []
      for a in angles:
        diff = abs(a - theta0)
        diff = min(diff, 2*np.pi - diff)
        slim_factor = 3.0
        r = avg_r * (0.7 + 0.6 * np.exp(-slim_factor * diff))
        r *= (0.8 + 0.4 * np.random.rand())
        radii.append(r)


      points = []
      for a, r in zip(angles, radii):
        x = cx + r * np.cos(a)
        y = cy + r * np.sin(a)
        points.append([int(np.clip(x, 0, W-1)), int(np.clip(y, 0, H-1))])

      polygon = np.array(points)
      mask = np.zeros((H, W), dtype=np.uint8)

      for y in range(H):
        xs = []
        for i in range(num_vertices):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i+1) % num_vertices]
            if y1 == y2:
                continue
            if (y >= min(y1,y2)) and (y < max(y1,y2)):
                x_int = x1 + (y - y1)*(x2 - x1)/(y2 - y1)
                xs.append(int(x_int))
        xs.sort()
        for i in range(0, len(xs), 2):
            if i+1 < len(xs):
                mask[y, xs[i]:xs[i+1]] = 1

      return torch.tensor(mask).float().unsqueeze(0)


    def __getitem__(self, idx):
        img_idx = idx // self.masks_per_image
        img = Image.open(self.img_files[img_idx]).convert("RGB")
        img = self.transform(img)
        C, H, W = img.shape
        mask = self._gen_random_irregular_mask(H, W)
        masked_img = img * (1 - mask)
        return masked_img, img, mask
    
    
class Stage1Dataset(Dataset):
    def __init__(self, root_dir, img_size=512, masks_per_image=100):
        self.img_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.img_size = img_size
        self.masks_per_image = masks_per_image

        self.transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: TF.rotate(x, 90)),
                transforms.Lambda(lambda x: TF.rotate(x, 180)),
                transforms.Lambda(lambda x: TF.rotate(x, 270)),
            ]),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.img_files) * self.masks_per_image

    def _gen_random_bbox(self):
        if random.random() < 0.5:
            aspect_ratio = random.uniform(1, 33)
        else:
            aspect_ratio = random.uniform(0.03, 1)

        area = random.uniform(0.1, 0.33) ** 2
        w = min(np.sqrt(area * aspect_ratio), 0.99)
        h = min(np.sqrt(area / aspect_ratio), 0.99)

        x1 = random.uniform(0.05, 0.99 - w)
        y1 = random.uniform(0.05, 0.99 - h)

        return torch.tensor([x1, y1, x1 + w, y1 + h], dtype=torch.float16)

    def __getitem__(self, idx):
        img_idx = idx // self.masks_per_image
        img = Image.open(self.img_files[img_idx]).convert("RGB")
        img = self.transform(img)

        bbox = self._gen_random_bbox()

        h, w = img.shape[1], img.shape[2]
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        mask = torch.zeros_like(img)
        mask[:, x1:x2, y1:y2] = 1

        masked_img = img * (1 - mask)

        return masked_img, img, bbox