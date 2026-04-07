import os, json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class PrecomputedCascadeDataset(Dataset):
    """
    Loads the precomputed .pt files containing:
        - z_cond [4,64,64]
        - target_img [3,512,512] in [-1,1]
    """
    def __init__(self, index_jsonl: str | Path):
        index_jsonl = Path(index_jsonl)
        if not index_jsonl.exists():
            raise FileNotFoundError(f"Index file not found: {index_jsonl}")
        
        self.items = []
        with open(index_jsonl, "r") as f:
            for line in f:
                path = json.loads(line)["pt"]
                if os.path.exists(path):
                    self.items.append(path)
                else:
                    print(f"[WARN] Missing file: {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        pack = torch.load(self.items[idx], map_location="cpu")
        return (
            pack["target_img"].float(),
            pack["z_cond"].to(torch.float16)
        )