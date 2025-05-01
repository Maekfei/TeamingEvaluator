import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class WeightedImputer(nn.Module):
    """
    v_{p,t} = Σ_m  w_m  ·  mean_{i∈N_p,t^m} u_{i,t}
    One scalar weight per metadata type (author, venue, keyword, …).
    """
    def __init__(self, meta_types):
        super().__init__()
        self.w = nn.ParameterDict({
            m: nn.Parameter(torch.tensor(1.0)) for m in meta_types
        })

    def forward(self, paper_id: int, year_idx: int,
                snapshots, embeddings):
        """
        Args
        ----
        paper_id    : int          index of paper node in snapshot `year_idx`
        year_idx    : int          position inside `snapshots` list
        snapshots   : List[HeteroData]
        embeddings  : List[ Dict[str, Tensor] ]
        Returns
        -------
        v_{p,t}     : Tensor [hidden_dim]   imputed embedding for paper p at year t
        """
        data  = snapshots[year_idx]
        embs  = embeddings[year_idx]
        device = embs['paper'].device

        # -------------------------------------------------- #
        #  Collect neighbours (authors, venue) **on GPU**    #
        #  using integer indexing only – no boolean mask.    #
        # -------------------------------------------------- #
        neighbors = {}

        # 1) authors ────────────────────────────────────────
        e_src, e_dst = data['author', 'writes', 'paper'].edge_index.to(device)
        # indices of edges whose dst is paper_id
        author_mask = (e_dst == paper_id).nonzero(as_tuple=False).view(-1)
        if author_mask.numel() > 0:
            author_ids = e_src.index_select(0, author_mask)
            neighbors['author'] = author_ids

        # 2) venue ──────────────────────────────────────────
        e_src, e_dst = data['paper', 'published_in', 'venue'].edge_index.to(device)
        venue_mask = (e_src == paper_id).nonzero(as_tuple=False).view(-1)
        if venue_mask.numel() > 0:
            venue_ids = e_dst.index_select(0, venue_mask)
            neighbors['venue'] = venue_ids

        # ------------- aggregate with type-wise weights -----------------
        if not neighbors:
            return torch.zeros_like(embs['paper'][paper_id])

        parts = []
        for ntype, ids in neighbors.items():
            if ids.numel() == 0:
                continue
            parts.append(self.w[ntype] * embs[ntype][ids].mean(dim=0))

        return torch.stack(parts, dim=0).sum(dim=0)