"""
Builds yearly PyG HeteroData snapshots from

    • paper_nodes_GNN_yearly.json.gz
    • tkg_embeddings_all_2024.npz

Author IDs, venue names and PubMed IDs are re-mapped to contiguous
indices that are *stable across years*.
"""

import gzip, json, os, pickle, numpy as np, torch
from torch_geometric.data import HeteroData
from tqdm import tqdm


# ------------- paths ------------------------------------------------------
PAPER_JSON   = '/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/' \
               'papernodes_remove0/paper_nodes_GNN_yearly.json.gz'
EMB_NPZ      = '/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/' \
               'tkg_embeddings_all_2024.npz'
CACHE_DIR    = 'data/yearly_snapshots'          # *.pt files go here
META_CACHE   = os.path.join(CACHE_DIR, 'mappings.pkl')   # id ↔ idx tables


# ------------- load the big embedding file --------------------------------
print('[dataset_builder] loading SPECTER 2 embeddings …')
with np.load(EMB_NPZ, mmap_mode='r') as npz:
    emb_matrix = npz['embeddings']       # mem-mapped (N, 768) float32
    emb_ids    = npz['ids'].astype(str)  # 1-to-1 list of PubMed IDs

id2embrow = {pid: i for i, pid in enumerate(emb_ids)}
EMB_DIM   = emb_matrix.shape[1]


# ------------- helpers: mapping tables ------------------------------------
def load_or_init_mappings():
    if os.path.isfile(META_CACHE):
        print('[dataset_builder] loading cached id-maps …')
        with open(META_CACHE, 'rb') as fh:
            return pickle.load(fh)

    # create from scratch --------------------------------------------------
    print('[dataset_builder] building id-maps …')
    with gzip.open(PAPER_JSON, 'rt', encoding='utf-8') as f:
        paper_json = json.load(f)

    aut2idx, ven2idx = {}, {}
    paper2idx, idx2paper = {}, []

    for pid, node in tqdm(paper_json.items(), desc='scan json'):
        # papers -----------------------------------------------------------
        if pid not in paper2idx:
            paper2idx[pid] = len(idx2paper)
            idx2paper.append(pid)
        # authors ----------------------------------------------------------
        for aid in node['neighbors']['author']:
            if aid not in aut2idx:
                aut2idx[aid] = len(aut2idx)
        # venues -----------------------------------------------------------
        ven = node['features']['Venue']
        if ven not in ven2idx:
            ven2idx[ven] = len(ven2idx)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(META_CACHE, 'wb') as fh:
        pickle.dump((paper2idx, aut2idx, ven2idx), fh)

    return paper2idx, aut2idx, ven2idx


PAPER2IDX, AUT2IDX, VEN2IDX = load_or_init_mappings()


# ------------- create one snapshot ---------------------------------------
def build_snapshot(up_to_year: int,
                   L: int = 5,
                   core_only_labels: bool = True) -> HeteroData:
    """
    Parameters
    ----------
    up_to_year : include papers published ≤ this year
    L          : prediction horizon (# yearly citation counts)
    core_only_labels : if True, loss is computed only for core papers
    """
    print(f'[dataset_builder]   ⇒ building snapshot ≤ {up_to_year}')
    with gzip.open(PAPER_JSON, 'rt', encoding='utf-8') as f:
        paper_json = json.load(f)

    # ---------------- collect node-level tensors --------------------------
    num_papers = sum(1 for v in paper_json.values()
                     if v['features']['PubYear'] <= up_to_year)
    num_authors_est = len(AUT2IDX)       # upper bound
    num_venues      = len(VEN2IDX)

    data = HeteroData()

    # ----- paper features -------------------------------------------------
    x_paper  = torch.zeros(num_papers, EMB_DIM, dtype=torch.float16)
    y_cit    = torch.zeros(num_papers, L,        dtype=torch.long)
    is_core  = torch.zeros(num_papers,           dtype=torch.bool)

    paper_idx_of = {}        # PubMed ID  → local index

    for pid, node in paper_json.items():
        year = node['features']['PubYear']
        if year > up_to_year:
            continue
        p_idx = len(paper_idx_of)
        paper_idx_of[pid] = p_idx


        # SPECTER 2 embedding  (falls back to zeros if missing)
        row = id2embrow.get(pid, None)
        if row is not None:
            x_paper[p_idx] = torch.from_numpy(emb_matrix[row]).astype(np.float16)

        # citation labels
        for l in range(1, L + 1):
            y_cit[p_idx, l - 1] = node['features'].get(
                f'yearly_citation_count_{l}', 0)

        # core flag
        if node['features']['is_core'] == 1:
            is_core[p_idx] = True

    data['paper'].x_title_emb   = x_paper
    data['paper'].y_citations   = y_cit
    data['paper'].is_core       = is_core       # ← used inside loss fn

    # ----- author & venue placeholders -----------------------------------
    data['author'].x = torch.randn(len(AUT2IDX), 50)   # small random vecs
    data['venue' ].x = torch.randn(num_venues, 50)

    # ---------------- edges ----------------------------------------------
    # 1) author ⟶ paper (“writes”)
    src, dst = [], []
    for pid, node in paper_json.items():
        if node['features']['PubYear'] > up_to_year:
            continue
        p_idx = paper_idx_of[pid]
        for aid in node['neighbors']['author']:
            src.append(AUT2IDX[aid])
            dst.append(p_idx)
    data['author', 'writes', 'paper'].edge_index = \
        torch.tensor([src, dst], dtype=torch.long)
    data['paper', 'written_by', 'author'].edge_index = \
        torch.tensor([dst, src], dtype=torch.long)

    # 2) paper ⟶ paper (“cites”)
    src, dst = [], []
    for pid, node in paper_json.items():
        if node['features']['PubYear'] > up_to_year:
            continue
        p_idx = paper_idx_of[pid]
        for ref in node['neighbors']['reference_papers']:
            if ref in paper_idx_of:                 # referenced paper ≤ year
                src.append(p_idx)
                dst.append(paper_idx_of[ref])
    data['paper', 'cites',    'paper'].edge_index = \
        torch.tensor([src, dst], dtype=torch.long)
    data['paper', 'cited_by', 'paper'].edge_index = \
        torch.tensor([dst, src], dtype=torch.long)

    # 3) paper ⟶ venue (“published_in”)
    src, dst = [], []
    for pid, node in paper_json.items():
        if node['features']['PubYear'] > up_to_year:
            continue
        p_idx = paper_idx_of[pid]
        v_idx = VEN2IDX[node['features']['Venue']]
        src.append(p_idx)
        dst.append(v_idx)
    data['paper', 'published_in', 'venue'].edge_index = \
        torch.tensor([src, dst], dtype=torch.long)
    data['venue', 'publishes', 'paper'].edge_index = \
        torch.tensor([dst, src], dtype=torch.long)

    # done ----------------------------------------------------------------
    return data