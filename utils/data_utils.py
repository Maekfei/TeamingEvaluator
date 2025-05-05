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
            snapshots.append(ensure_x_field(torch.load(f, weights_only=False)))
            continue

        from utils.dataset_builder import build_snapshot
        if not generate_if_missing:
            raise FileNotFoundError(f"Snapshot {f} not found.")

        print(f'[data_utils] snapshot {f} not found – generating on the fly …')
        snap = build_snapshot(y, L=L)
        snapshots.append(ensure_x_field(snap))

        if save_generated:
            os.makedirs(os.path.dirname(f), exist_ok=True)
            torch.save(snap, f)

    return snapshots