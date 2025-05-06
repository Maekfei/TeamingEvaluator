"""
Builds yearly PyG HeteroData snapshots from

    • paper_nodes_GNN_yearly.json.gz
    • tkg_embeddings_all_2024.npz

Author IDs, venue names and PubMed IDs are re-mapped to contiguous
indices that are *stable across years*.
"""

import gzip, json, os, pickle, torch
import numpy as np
from torch_geometric.data import HeteroData
from tqdm import tqdm


# ------------- paths ------------------------------------------------------
PAPER_JSON = '/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/' \
'papernodes_remove0/paper_nodes_GNN_yearly.json.gz'
# EMB_NPZ = '/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/' \
# 'tkg_embeddings_all_2024.npz'
EMB_NPZ = '/data/jx4237data/GNNteamingEvaluator/TeamingEvaluator/data_examine/output_npz/OpenAI_paper_embeddings.npz'
CACHE_DIR    = 'data/yearly_snapshots'          # *.pt files go here
META_CACHE   = os.path.join(CACHE_DIR, 'mappings.pkl')   # id ↔ idx tables

# ------------- load the paper JSON file ---------------------------------
print('[dataset_builder] loading paper JSON …')
with gzip.open(PAPER_JSON, 'rt', encoding='utf-8') as f:
    paper_json = json.load(f)

# ------------- load the big embedding file --------------------------------
print('[dataset_builder] loading SPECTER 2 embeddings …')
with np.load(EMB_NPZ, mmap_mode='r') as npz:
    emb_matrix = npz['embeddings']       # mem-mapped (N, 768) float32
    emb_ids    = npz['ids'].astype(str)  # 1-to-1 list of PubMed IDs
print(f'[dataset_builder]   ⇒ {emb_matrix.shape[0]} embeddings')
print(f'[dataset_builder]   ⇒ {emb_matrix.shape[1]} dimensions')
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

    aut2idx, ven2idx = {}, {}
    paper2idx, idx2paper = {}, [] # list as mapping table

    for pid, node in tqdm(paper_json.items(), desc='scan json'):
        # papers -----------------------------------------------------------
        if node['features']['is_core'] == 0: # important, only choose around ~ 9k papers
            continue
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
    # print the sizes of the mappings (papers, authors, venues)
    print(f'[dataset_builder]   ⇒ {len(paper2idx):,} papers')
    print(f'[dataset_builder]   ⇒ {len(aut2idx):,} authors')
    print(f'[dataset_builder]   ⇒ {len(ven2idx):,} venues')
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


    # ---------------- collect node-level tensors --------------------------
    num_papers = sum(1 for v in paper_json.values()
                     if v['features']['PubYear'] <= up_to_year and
                        v['features']['is_core'] == 1) # important, only choose around ~ 9k papers
    num_authors_est = len(AUT2IDX)       # upper bound
    num_venues      = len(VEN2IDX)

    data = HeteroData()


    # ----- paper features -------------------------------------------------
    x_paper  = torch.zeros(num_papers, EMB_DIM, dtype=torch.float32)
    y_cit    = torch.zeros(num_papers, L,        dtype=torch.long)
    is_core  = torch.zeros(num_papers,           dtype=torch.bool)

    paper_idx_of = {}        # PubMed ID  → local index
    for pid, node in paper_json.items():
        year = node['features']['PubYear']
        if year > up_to_year or node['features']['is_core'] == 0:
            continue
        p_idx = len(paper_idx_of)
        paper_idx_of[pid] = p_idx # local index


        # SPECTER 2 embedding  (falls back to zeros if missing); here it will be 768 dim
        row = id2embrow.get(pid, None)
        if row is not None:
            x_paper[p_idx] = torch.from_numpy(emb_matrix[row])

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

    # ---------------------------------------------------------------
    # Author / venue embeddings (pre-computed JSON files)
    # ---------------------------------------------------------------
    EMB_DIR = '/data/jx4237data/GNNteamingEvaluator/TeamingEvaluator/data_examine/output_embeddings_yearly'          # ← adjust if necessary
    fn_author = os.path.join(EMB_DIR,
                             f'author_embedding_{up_to_year}.json')
    fn_venue  = os.path.join(EMB_DIR,
                             f'venue_embedding_{up_to_year}.json')

    # ----------------------------------------------------------------
    # 1) tensors initialised with zeros  (unknown ids keep zero vector)
    # ----------------------------------------------------------------
    x_author = torch.zeros(len(AUT2IDX),  EMB_DIM, dtype=torch.float32)
    x_venue  = torch.zeros(len(VEN2IDX),  EMB_DIM, dtype=torch.float32)

    # helper to copy a Python list -> torch row
    def _copy_into(row_idx: int, vec: list, dest: torch.Tensor):
        try:
            dest[row_idx] = torch.tensor(vec, dtype=dest.dtype)
        except Exception as e:
            # silently ignore malformed / wrong-dim vectors
            print(f'  [warn] could not copy embedding for row {row_idx}: {e}')

    # ----------------------------------------------------------------
    # 2) load author embeddings
    # ----------------------------------------------------------------
    if os.path.isfile(fn_author):
        with open(fn_author, 'r', encoding='utf-8') as fh:
            auth_json = json.load(fh)

        for aid, vec in auth_json.items():
            a_idx = AUT2IDX.get(aid)
            if a_idx is not None:
                _copy_into(a_idx, vec, x_author)
    else:
        print(f'  [warn] author embedding file not found: {fn_author}')

    # ----------------------------------------------------------------
    # 3) load venue embeddings
    # ----------------------------------------------------------------
    if os.path.isfile(fn_venue):
        with open(fn_venue, 'r', encoding='utf-8') as fh:
            ven_json = json.load(fh)

        for ven, vec in ven_json.items():
            v_idx = VEN2IDX.get(ven)
            if v_idx is not None:
                _copy_into(v_idx, vec, x_venue)
    else:
        print(f'  [warn] venue embedding file not found: {fn_venue}')

    # ----------------------------------------------------------------
    # 4) attach to the HeteroData object
    # ----------------------------------------------------------------
    data['author'].x = x_author
    data['venue' ].x = x_venue

    # ---------------- edges ----------------------------------------------
    # 1) author ⟶ paper (“writes”)
    src, dst = [], []
    for pid, node in paper_json.items():
        if node['features']['PubYear'] > up_to_year or \
           node['features']['is_core'] == 0: # important, only choose around ~ 9k papers
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
        if node['features']['PubYear'] > up_to_year or \
           node['features']['is_core'] == 0: # important, only choose around ~ 9k papers
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
        if node['features']['PubYear'] > up_to_year or \
           node['features']['is_core'] == 0: # important, only choose around ~ 9k papers
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