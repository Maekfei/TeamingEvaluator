import torch
import torch.nn as nn
from .rgcn_encoder import RGCNEncoder
from .imputer import WeightedImputer
from .impact_rnn import ImpactRNN
from typing import List
from utils.metrics import rmsle_vec, male_vec


class ImpactModel(nn.Module):
    """
    Puts all three modules together and computes joint loss:
        J = L_pred  +  β Σ_t L_time(t,t+1)
    """
    def __init__(
        self,
        metadata,
        in_dims,
        hidden_dim=128,
        beta=1e-3,
        horizons=(1, 2, 3, 4, 5),
        meta_types=("author", "venue"),
    ):
        super().__init__()
        self.encoder = RGCNEncoder(metadata, in_dims, hidden_dim)
        self.imputer = WeightedImputer(meta_types)
        self.generator = ImpactRNN(hidden_dim)
        self.beta = beta
        self.horizons = torch.tensor(horizons, dtype=torch.float32)
        self.eps = 1.0                    # +1 for zero-citation papers

    # -----------------------------------------------------------------
    def forward(self, snapshots, years_train):
        """
        Train forward pass iterating over years in years_train.
        snapshots : list[HeteroData] aligned with chronological order.
        years_train: iterable of indices (ints)
        Returns:
            loss  (scalar)
            log   (dict for printing)
        """
        device = next(self.parameters()).device
        self.horizons = self.horizons.to(device, non_blocking=True)


        # 1) encode every snapshot once
        embeddings = []
        for data in snapshots:
            embeddings.append(self.encoder(data))

        # -------- temporal smoothing regulariser --------------------
        l_time = []
        for t in range(len(snapshots) - 1):
            common_papers = torch.arange(
                min(embeddings[t]['paper'].size(0),
                    embeddings[t + 1]['paper'].size(0)),
                device=device,
            )
            diff = (
                embeddings[t]['paper'][common_papers]
                - embeddings[t + 1]['paper'][common_papers]
            )
            l_time.append((diff ** 2).sum(dim=1).mean())
        l_time = torch.stack(l_time).mean() if l_time else torch.tensor(0.0)

        # -------- prediction loss over all new papers ----------------
        l_pred = []
        for t in years_train:
            data = snapshots[t]
            y_true = data['paper'].y_citations.to(device).float()  # [N, L]
            N, L = y_true.shape
            # Impute one vector v_{p, t}  for each paper in current snapshot
            v_t = torch.stack(
                [
                    self.imputer(pid, t, snapshots, embeddings)
                    for pid in range(N)
                ],
                dim=0,
            )                                           # [N, hidden]
            # For each paper we need sequence V_p of length T'<=5 (years before)
            # Here, for simplicity we replicate v_t for 5 tokens
            V_p = v_t.unsqueeze(1).repeat(1, 5, 1)      # [N, 5, hidden]
            eta, mu, sigma = self.generator(V_p)
            y_hat_cum = self.generator.predict_cumulative(
                self.horizons.to(device), eta, mu, sigma
            )                                           # [N, L]
            # convert to per-year citation counts
            y_hat = torch.cat(
                [y_hat_cum[:, 0:1], y_hat_cum[:, 1:] - y_hat_cum[:, :-1]],
                dim=1,
            )
            loss = ((torch.log1p(y_true + self.eps) - torch.log1p(y_hat)) ** 2).mean()
            l_pred.append(loss)
        l_pred = torch.stack(l_pred).mean()

        loss = l_pred + self.beta * l_time

        log = dict(
            L_pred=l_pred.item(),
            L_time=l_time.item(),
            Loss=loss.item(),
        )
        return loss, log

    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, snapshots, years_test):
        device = self.horizons.device
        self.horizons = self.horizons.to(device)

        embeddings = [self.encoder(data) for data in snapshots]

        males, rmsles = [], []
        for t in years_test:
            data = snapshots[t]
            y_true = data['paper'].y_citations.to(device).float()
            N, L = y_true.shape

            v_t = torch.stack(
                [self.imputer(pid, t, snapshots, embeddings) for pid in range(N)],
                dim=0,
            )
            V_p = v_t.unsqueeze(1).repeat(1, 5, 1)
            eta, mu, sigma = self.generator(V_p)
            y_hat_cum = self.generator.predict_cumulative(
                self.horizons.to(device), eta, mu, sigma
            )
            y_hat = torch.cat(
                [y_hat_cum[:, 0:1], y_hat_cum[:, 1:] - y_hat_cum[:, :-1]],
                dim=1,
            )

            males.append(male_vec(y_true, y_hat))
            rmsles.append(rmsle_vec(y_true, y_hat))

        male = torch.stack(males).mean(0)    # [L]
        rmsle = torch.stack(rmsles).mean(0)
        return male.cpu(), rmsle.cpu()