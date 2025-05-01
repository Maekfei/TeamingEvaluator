import os, torch
from torch_geometric.data import HeteroData


def ensure_x_field(data):
    # If a node type has no .x but has .x_title_emb, copy it.
    if 'paper' in data.node_types and not hasattr(data['paper'], 'x'):
        data['paper'].x = data['paper'].x_title_emb
    return data

def load_snapshots(path_pattern, years, emb_dim=128, L=5,
                   generate_if_missing=True, save_generated=True):
    """
    Returns list[HeteroData] for the requested years.
    If the file path_pattern.format(year) does not exist and
    generate_if_missing=True, a random snapshot is produced.
    """
    snapshots = []  # list of HeteroData objects for each year
    for y in years:
        f = path_pattern.format(y) # e.g. "data/raw/G_{}.pt".format(y)
        if os.path.isfile(f):
            snapshots.append(ensure_x_field(torch.load(f)))

        elif generate_if_missing:
            snap = build_dummy_snapshot(emb_dim=emb_dim, L=L)
            snapshots.append(ensure_x_field(snap))
            if save_generated:
                os.makedirs(os.path.dirname(f), exist_ok=True)
                torch.save(snap, f)
        else:
            raise FileNotFoundError(f"Snapshot {f} not found.")
    return snapshots


# -------------- below: optional toy data generator -------------------
def build_dummy_snapshot(num_papers=100, num_authors=50, num_venues=10,
                         emb_dim=128, L=5):
    """
    Creates a small random HeteroData object – useful for debugging.
    """
    data = HeteroData() 

    # Nodes
    data['author'].x = torch.randn(num_authors, emb_dim)  # num_authors x emb_dim
    data['paper'].x_title_emb = torch.randn(num_papers, 768)
    data['venue'].x = torch.randn(num_venues, emb_dim)

    # Edges
    # authors write papers
    src = torch.randint(0, num_authors, (num_papers * 2,)) # src = authors
    dst = torch.randint(0, num_papers, (num_papers * 2,)) # dst = papers
    data['author', 'writes', 'paper'].edge_index = torch.stack([src, dst])
    data['paper', 'written_by', 'author'].edge_index = torch.stack([dst, src])

    # citations
    src = torch.randint(0, num_papers, (num_papers * 3,))
    dst = torch.randint(0, num_papers, (num_papers * 3,))
    data['paper', 'cites', 'paper'].edge_index = torch.stack([src, dst])
    data['paper', 'cited_by', 'paper'].edge_index = torch.stack([dst, src])

    # publication
    src = torch.randint(0, num_papers, (num_papers,))
    dst = torch.randint(0, num_venues, (num_papers,))
    data['paper', 'published_in', 'venue'].edge_index = torch.stack([src, dst])
    data['venue', 'publishes', 'paper'].edge_index = torch.stack([dst, src])

    # labels (future citations)
    data['paper'].y_citations = torch.randint(0, 50, (num_papers, L))
    return data