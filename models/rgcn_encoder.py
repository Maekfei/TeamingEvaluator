from typing import Dict
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv


class RGCNEncoder(nn.Module):
    """
    Relation-aware GCN used for every yearly snapshot.
    All weight matrices are shared across years – simply reuse one
    instance of this module for all time steps.
    """
    def __init__(
        self,
        metadata: Dict,
        in_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.lins = nn.ModuleDict()
        # project every node type to hidden_dim (if needed)
        for ntype, in_dim in in_dims.items():
            if in_dim != hidden_dim:
                self.lins[ntype] = nn.Linear(in_dim, hidden_dim, bias=False)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                { etype: SAGEConv(hidden_dim, hidden_dim)
                    for etype in metadata[1]         # edge types
                },
                aggr="mean",
            )
            self.convs.append(conv)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins.values():
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        """
        Args
        ----
        data : torch_geometric.data.HeteroData
            must provide   data[edge_type].edge_index   and node features.
        Returns
        -------
        x_dict : Dict[str, Tensor]   node_type -> embeddings
        """
        # 1) initial linear projection (shared across years)
        x_dict = {}
        for ntype in data.node_types:
            if hasattr(data[ntype], "x"):
                h = data[ntype].x
            elif hasattr(data[ntype], "x_title_emb"):        # fallback for papers
                h = data[ntype].x_title_emb
            else:
                raise KeyError(f"No feature tensor (.x) for node type '{ntype}'")

            if ntype in self.lins:                           # project if needed
                h = self.lins[ntype](h)
            x_dict[ntype] = h

        # 2) stacked relation-specific GCN layers
        for hetero_layer in self.convs:                      # type: HeteroConv
            out_dict = {k: [] for k in x_dict.keys()}        # collect msg per dst

            for etype_str, subconv in hetero_layer.convs.items():
                # etype_str e.g.  'author__writes__paper'
                src, rel, dst = etype_str.split("__")
                edge_index = data[(src, rel, dst)].edge_index.to(x_dict[src].device)
                if (edge_index.numel() == 0
                    or x_dict[src].size(0) == 0
                    or x_dict[dst].size(0) == 0):
                    continue
                    

                # dummy relation-id vector (because num_relations = 1)
                # edge_type = edge_index.new_zeros(edge_index.size(1))

                # message passing
                # out = subconv((x_dict[src], x_dict[dst]), edge_index, edge_type)
                out = subconv((x_dict[src], x_dict[dst]), edge_index)
                out_dict[dst].append(out)

            # mean-aggregate incoming relations for every destination type
            x_dict = {
                ntype: self.dropout(torch.relu(torch.stack(msgs, 0).mean(0)))
                if len(msgs) > 0 else x_dict[ntype]           # keep old if isolated
                for ntype, msgs in out_dict.items()
            }

        return x_dict