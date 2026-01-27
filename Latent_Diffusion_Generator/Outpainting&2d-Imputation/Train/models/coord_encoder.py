import torch.nn as nn

class CoordEncoder(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU()
        )

    def forward(self, bbox):
        return self.net(bbox)
