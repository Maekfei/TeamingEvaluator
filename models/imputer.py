import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class WeightedImputer(nn.Module):
    """
    v_{p,t} = Σ_m  w_m  ·  mean_{i∈N_p,t^m} u_{i,t}
    One scalar weight per metadata type (author, venue, reference, …).
    """
    def __init__(self, meta_types): # meta_types: list of metadata types (e.g. ['author', 'venue'])
        super().__init__()
        self.w = nn.ParameterDict({
            m: nn.Parameter(torch.tensor(1.0)) for m in meta_types
        })

    @staticmethod
    def collect_neighbours(data, paper_id: int, device):
        """
        Collect neighbour indices *in the publication year* snapshot.

        Returns a dict
            { 'author': LongTensor,        # authors of the paper
              'venue' : LongTensor,        # venue of the paper
              'paper' : LongTensor }       # references (= cited papers)
        """
        neighbours = {}

        # 1) authors --------------------------------------------------------
        src, dst = data['author', 'writes', 'paper'].edge_index.to(device)
        mask = (dst == paper_id).nonzero(as_tuple=False).view(-1)
        if mask.numel():
            neighbours['author'] = src.index_select(0, mask)

        # 2) venue ----------------------------------------------------------
        src, dst = data['paper', 'published_in', 'venue'].edge_index.to(device)
        mask = (src == paper_id).nonzero(as_tuple=False).view(-1)
        if mask.numel():
            neighbours['venue'] = dst.index_select(0, mask)

        # 3) references (citations) ----------------------------------------
        src, dst = data['paper', 'cites', 'paper'].edge_index.to(device)
        mask = (src == paper_id).nonzero(as_tuple=False).view(-1)
        if mask.numel():
            neighbours['paper'] = dst.index_select(0, mask)

        return neighbours
    # ======================================================================
    # ======================================================================




    def forward(
        self,
        paper_id: int | None,
        year_idx: int,
        snapshots,
        embeddings,
        predefined_neigh: dict[str, torch.Tensor] | None = None,
    ):
        """
        Args
        ----
        paper_id          : index of paper *in its publication year* snapshot.
                            Ignored when `predefined_neigh` is given.
        year_idx          : index of the snapshot we want to impute for
                            (t-1, t-2, …).
        snapshots         : list[HeteroData]
        embeddings        : list[dict] – output of the encoder for every year
        predefined_neigh  : optional neighbour dict produced by
                            `collect_neighbours`.  Needed because the paper
                            itself is not present in earlier graphs.

        Returns
        -------
        Tensor [hidden_dim] – imputed embedding v_{p, year_idx}
        """
        data = snapshots[year_idx]
        embs = embeddings[year_idx]
        device = embs['paper'].device

        # decide which neighbour set to use -------------------------------
        if predefined_neigh is not None:
            neighbours = predefined_neigh
        else:
            neighbours = self.collect_neighbours(data, paper_id, device)

        # ----- nothing to aggregate --------------------------------------
        if not neighbours:
            return torch.zeros(embs['paper'].size(-1), device=device)

        # ----- weighted average  -----------------------------------------
        parts = []
        for ntype, ids in neighbours.items():
            # some neighbour ids may not exist yet in an earlier snapshot
            ids = ids[ids < embs[ntype].size(0)]
            if ids.numel() == 0:
                continue
            parts.append(self.w[ntype] * embs[ntype][ids].mean(dim=0))

        if len(parts) == 0:
            return torch.zeros(embs['paper'].size(-1), device=device)

        return torch.stack(parts, dim=0).sum(dim=0)