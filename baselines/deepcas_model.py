# baselines/deepcas_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.metrics import male_vec, rmsle_vec

class DeepCasModel(nn.Module):
    """
    Highly simplified DeepCas skeleton.

    Only the *training loop* and *evaluation API* are provided.
    You need to:
    - implement random-walk generator on the DHIN snapshot
    - implement the DeepCas encoder (CNN/LSTM over walks)
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Identity()          # TODO: replace by DeepCas encoder
        self.regressor = nn.Sequential(       # 3-layer MLP → 5 outputs
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    # ------------------------------------------------------------------
    def forward(self, batch):
        """
        batch: output of random-walk loader – implement yourself.
        Should return Tensor [B, hidden_dim] before regressor.
        """
        z = self.encoder(batch)               # TODO
        return self.regressor(z)              # [B, 5]

    # ------------------------------------------------------------------
    def train_one_epoch(self, loader, optim, device):
        self.train()
        total = 0
        for batch, target in loader:          # target  [B, 5]
            batch, target = batch.to(device), target.to(device).float()
            optim.zero_grad()
            pred = self(batch)
            loss = ((pred - target) ** 2).mean()
            loss.backward()
            optim.step()
            total += loss.item() * batch.size(0)
        return total / len(loader.dataset)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader, device):
        self.eval()
        males, rmsles = [], []
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device).float()
            pred = self(batch)
            males .append(male_vec (target, pred))
            rmsles.append(rmsle_vec(target, pred))
        return (torch.stack(males ).mean(0).cpu(),
                torch.stack(rmsles).mean(0).cpu())