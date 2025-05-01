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
    where L_pred is the prediction loss and L_time(t,t+1) is the temporal
    smoothing regulariser.
    β  controls the degree of temporal smoothing.
    """
    def __init__(
        self,
        metadata,
        in_dims,
        hidden_dim=128,
        beta=.5, # regularization parameter (temporal smoothing regularizer of the temporal graph, make sure the same papers are not too different in the two consecutive years)
        horizons=(1, 2, 3, 4, 5), # [1, 2, 3, 4, 5] yearly citation counts
        meta_types: tuple[str, ...] = ("author", "venue", "paper"),
    ):
        super().__init__()
        self.encoder = RGCNEncoder(metadata, in_dims, hidden_dim) # output: keys ('author', 'paper', 'venue'), values: embeddings. (num_nodes_of_that_type, hidden_dim)
        self.imputer = WeightedImputer(meta_types) # impute to update the new paper embedding.
        self.generator = ImpactRNN(hidden_dim)
        self.beta = beta
        self.hidden_dim = hidden_dim

        self.history   = len(horizons)      # number of years fed to the GRU
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

        # -------- temporal smoothing regulariser -------------------- # make sure the same papers, authors, venues are not too different in the two consecutive years
        l_time = []
        node_types_to_regularize = ['paper', 'author', 'venue']  # Add other node types if needed

        for node_type in node_types_to_regularize:
            l_time_ntype = []
            for t in range(len(snapshots) - 1):
                emb_t = embeddings[t].get(node_type)
                emb_t_plus_1 = embeddings[t + 1].get(node_type)

                if emb_t is not None and emb_t_plus_1 is not None:
                    common_nodes = torch.arange(
                        min(emb_t.size(0), emb_t_plus_1.size(0)),
                        device=device,
                    )
                    diff = emb_t[common_nodes] - emb_t_plus_1[common_nodes]
                    l_time_ntype.append((diff ** 2).sum(dim=1).mean())

            if l_time_ntype:
                l_time.append(torch.stack(l_time_ntype).mean())

        l_time = torch.stack(l_time).mean() if l_time else torch.tensor(0.0)

        # -------- prediction loss over all new papers ----------------
        l_pred = []
        for t in years_train: # year t.
            data = snapshots[t]
            y_true = data['paper'].y_citations.to(device).float()  # [N, L]
            N, _ = y_true.shape

            # ----------------------------------------------------------
            # 3.1 cache neighbours of every paper in its publication yr
            # ----------------------------------------------------------
            neigh_cache = [
                self.imputer.collect_neighbours(data, pid, device)
                for pid in range(N)
            ]

            # ----------------------------------------------------------
            # 3.2 build sequence  [t-1, …, t-history]
            # ----------------------------------------------------------
            seq_steps = []
            for k in range(1, self.history + 1):
                yr = t - k
                if yr < 0:
                    # zero-padding for years before dataset starts
                    seq_k = torch.zeros(
                        N, self.hidden_dim, device=device)
                else:
                    seq_k = torch.stack(
                        [
                            self.imputer(
                                pid,                     # dummy
                                yr,
                                snapshots,
                                embeddings,
                                predefined_neigh=neigh_cache[pid],
                            )
                            for pid in range(N)
                        ],
                        dim=0,
                    )                                   # [N, hidden]
                seq_steps.append(seq_k)

            V_p = torch.stack(seq_steps, dim=1)         # [N, T, hidden]

            # 3.3 generate citation distribution -----------------------
            eta, mu, sigma = self.generator(V_p)
            y_hat_cum = self.generator.predict_cumulative(
                self.horizons.to(device), eta, mu, sigma
            )                                           # [N, L]

            # convert cumulative → yearly counts -----------------------
            y_hat = torch.cat(
                [y_hat_cum[:, :1], y_hat_cum[:, 1:] - y_hat_cum[:, :-1]],
                dim=1,
            )

            loss = ((torch.log1p(y_true + self.eps) -
                     torch.log1p(y_hat)) ** 2).mean()
            l_pred.append(loss)

        l_pred = torch.stack(l_pred).mean()
        loss   = l_pred + self.beta * l_time

        log = dict(L_pred=l_pred.item(),
                   L_time=l_time.item(),
                   Loss  =loss.item())
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