from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F # Import for LeakyReLU
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
        hidden_dim: int = 32,
        num_layers: int = 2, # followed hints paper
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim # Store hidden_dim
        self.dropout_layer = nn.Dropout(dropout) # Renamed to avoid conflict with method name

        self.lins = nn.ModuleDict()
        self.norms_initial = nn.ModuleDict() # For initial projection normalization
        # project every node type to hidden_dim (if needed)
        for ntype, in_dims_ in in_dims.items():
            if in_dims_ != hidden_dim:
                # Changed bias to True to allow for non-zero output from zero input
                self.lins[ntype] = nn.Linear(in_dims_, hidden_dim, bias=True)
            self.norms_initial[ntype] = nn.BatchNorm1d(hidden_dim) # Add BatchNorm for initial projection

        # Add a linear layer for projecting social features to hidden_dim
        self.social_lin = nn.Linear(3, hidden_dim)  # 3 social features
        self.norms_social = nn.BatchNorm1d(hidden_dim)

        self.convs = nn.ModuleList()
        self.norms_conv = nn.ModuleDict() # For normalization after each conv layer
        for i in range(num_layers):
            conv = HeteroConv(
                { etype: SAGEConv(hidden_dim, hidden_dim)
                    for etype in metadata[1] 
                },
                aggr="mean",
            )
            self.convs.append(conv)
            # Add a BatchNorm layer for each node type after each convolution
            # This applies to the *output* of the convolution for each node type
            for ntype in metadata[0]: # Assuming metadata[0] provides all node types
                 self.norms_conv[f'layer_{i}_{ntype}'] = nn.BatchNorm1d(hidden_dim)


        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins.values():
            lin.reset_parameters()
        for norm in self.norms_initial.values():
            norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms_conv.values():
            norm.reset_parameters()

    def forward(self, data):
        """
        Args
        ----
        data : torch_geometric.data.HeteroData
            must provide data[edge_type].edge_index and node features.
        Returns
        -------
        x_dict : Dict[str, Tensor]   node_type -> embeddings
        """
        # 1) initial linear projection (shared across years)
        x_dict = {}
        for ntype in data.node_types:
            h = None
            if hasattr(data[ntype], "x"):
                h = data[ntype].x
            elif hasattr(data[ntype], "x_title_emb"): # fallback for papers
                h = data[ntype].x_title_emb
            else:
                raise KeyError(f"No feature tensor (.x or .x_title_emb) for node type '{ntype}'")

            if h is None: # Handle case where no features are found
                # If no features, initialize with zeros but allow the model to learn.
                # This case should ideally be caught by the KeyError above if no features are present.
                # However, if features are present but all zeros, then the linear layer with bias=True
                # and subsequent normalization will help.
                h = torch.zeros(data[ntype].num_nodes, self.hidden_dim, device=h.device) 
            
            if ntype in self.lins: # project if needed
                h = self.lins[ntype](h)
            
            # Apply initial normalization for all node types
            if h.numel() > 0: # Only apply BatchNorm if there are nodes
                h = self.norms_initial[ntype](h)
            x_dict[ntype] = h

        # Add social embedding for authors
        if 'author' in data.node_types and hasattr(data['author'], 'x_social'):
            h_social = self.social_lin(data['author'].x_social)
            if h_social.numel() > 0:
                h_social = self.norms_social(h_social)
            x_dict['author_social'] = h_social

        for i, hetero_layer in enumerate(self.convs): # type: HeteroConv
            out_dict = {k: [] for k in x_dict.keys()} # collect msg per dst

            for etype_str, subconv in hetero_layer.convs.items():
                # Handle different etype_str formats (tuple vs. string)
                if isinstance(etype_str, str):
                    src, rel, dst = etype_str.split('__')
                else:
                    src, rel, dst = etype_str # Assume it's already a tuple (src, rel, dst)
                               
                # Ensure edge_index exists and is not empty before processing
                if (src, rel, dst) not in data.edge_types:
                    continue # Skip if this edge type does not exist in the current graph
                    
                edge_index = data[(src, rel, dst)].edge_index.to(x_dict[src].device)

                # Skip message passing if any involved tensor is empty
                if (edge_index.numel() == 0
                    or x_dict[src].numel() == 0 # Check numel for empty tensors
                    or x_dict[dst].numel() == 0):
                    continue
                    
                # message passing
                out = subconv((x_dict[src], x_dict[dst]), edge_index)
                out_dict[dst].append(out)

            # mean-aggregate incoming relations for every destination type
            # Use LeakyReLU instead of ReLU to prevent dying neurons
            # Apply BatchNorm after aggregation and activation for each node type
            new_x_dict = {}
            for ntype, msgs in out_dict.items():
                if len(msgs) > 0:
                    h = torch.stack(msgs, 0).mean(0)
                    h = self.dropout_layer(F.leaky_relu(h, negative_slope=0.01))
                    
                    # Apply BatchNorm for the current layer and node type
                    if h.numel() > 0: # Ensure tensor is not empty
                        h = self.norms_conv[f'layer_{i}_{ntype}'](h)
                    new_x_dict[ntype] = h
                else:
                    new_x_dict[ntype] = x_dict[ntype] # keep old if isolated
            x_dict = new_x_dict
        
        return x_dict
