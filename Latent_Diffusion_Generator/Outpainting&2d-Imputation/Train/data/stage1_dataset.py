import os, random, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


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
