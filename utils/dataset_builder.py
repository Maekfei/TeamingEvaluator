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
# PAPER_JSON = 'data/paper_nodes_GNN_yearly.json.gz'
PAPER_JSON = '/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_GNN_yearly_9_year_citation_counts.json.gz'
EMB_NPZ = '/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/' \
'tkg_embeddings_all_2024.npz'
# # EMB_NPZ = 'data_examine/output_npz_openai/paper_embeddings_768_OpenAI.npz'
# CACHE_DIR    = 'data/yearly_snapshots_specter2'          # *.pt files go here
# CACHE_DIR    = 'data/yearly_snapshots_oai'          # *.pt files go here

CACHE_DIR    = 'data/yearly_snapshots_specter2_starting_from_year_1'          # *.pt files go here



META_CACHE   = os.path.join(CACHE_DIR, 'mappings_specter2_starting_from_year_1.pkl')   # id ↔ idx tables

# ------------- load the paper JSON file ---------------------------------
print('[dataset_builder] loading paper JSON …')
with gzip.open(PAPER_JSON, 'rt', encoding='utf-8') as f:
    paper_json = json.load(f)

# ------------- load the big embedding file --------------------------------
print('[dataset_builder] loading OAI embeddings …')
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

ACTIVE_PAPERS  : list[str] = []
ACTIVE_AUTHS   : list[str] = []
ACTIVE_VENUES  : list[str] = []


PAPER_LIDX_OF: dict[str, int] = {}      # global id  -> local contiguous id
AUTH_LIDX_OF : dict[str, int] = {}
VEN_LIDX_OF  : dict[str, int] = {}

def _assign_local_id(gid2lidx: dict, active_list: list, gid: str | int):
    """
    Helper – gives every *global* node id a *local* contiguous index
    that never changes once it has been assigned.
    """
    if gid not in gid2lidx:
        gid2lidx[gid] = len(active_list)
        active_list.append(gid)
    return gid2lidx[gid]

# ------------- create one snapshot ---------------------------------------
def build_snapshot(up_to_year: int, L: int = 5) -> HeteroData:
    print(f'[dataset_builder]   ⇒ building snapshot ≤ {up_to_year}')

    # --------------------------------------------------------------- #
    # 1) register every node that exists up-to this year              #
    # --------------------------------------------------------------- #
    paper_ids = [pid for pid, n in paper_json.items()
                 if n['features']['is_core'] == 1
                 and n['features']['PubYear'] <= up_to_year]

    for pid in sorted(paper_ids, key=lambda p: PAPER2IDX[p]):   # deterministic
        _assign_local_id(PAPER_LIDX_OF, ACTIVE_PAPERS, pid)

        n = paper_json[pid]
        for aid in n['neighbors']['author']:
            _assign_local_id(AUTH_LIDX_OF,  ACTIVE_AUTHS,  aid)
        ven = n['features']['Venue']
        _assign_local_id(VEN_LIDX_OF,       ACTIVE_VENUES, ven)

    num_papers  = len(ACTIVE_PAPERS)
    num_authors = len(ACTIVE_AUTHS)
    num_venues  = len(ACTIVE_VENUES)

    # --------------------------------------------------------------- #
    # 2)  build tensors (only existing rows, no zero-padding)         #
    # --------------------------------------------------------------- #
    data = HeteroData()

    # -------- papers ------------------------------------------------
    x_paper = torch.zeros(num_papers, EMB_DIM)
    y_cit   = torch.zeros(num_papers, L, dtype=torch.long)
    is_core = torch.zeros(num_papers,     dtype=torch.bool)
    y_year  = torch.zeros(num_papers,     dtype=torch.long)

    for pid in paper_ids:
        p = PAPER_LIDX_OF[pid]
        node = paper_json[pid]

        row = id2embrow.get(pid)
        if row is not None:
            x_paper[p] = torch.from_numpy(emb_matrix[row])

        for l in range(2, L + 2): # change here to starting from the year after the publication year from (1, L + 1) to (2, L + 2) , L = 5
            y_cit[p, l-2] = node['features'].get(f'yearly_citation_count_{l}', 0)

        is_core[p] = True
        y_year[p]  = node['features']['PubYear']

    data['paper'].x_title_emb = x_paper
    data['paper'].y_citations = y_cit
    data['paper'].is_core     = is_core
    data['paper'].y_year      = y_year

    # -------- authors / venues -------------------------------------
    x_author = torch.zeros(num_authors, EMB_DIM)
    x_venue  = torch.zeros(num_venues,  EMB_DIM)

    def _copy(idx: int, vec: list, dest: torch.Tensor):
        try:
            dest[idx] = torch.tensor(vec, dtype=dest.dtype)
        except Exception as e:
            print(f'[warn] could not copy embedding row {idx}: {e}')
    # EMB_DIR = '/data/jx4237data/GNNteamingEvaluator/TeamingEvaluator/data_examine/output_embeddings_yearly_oai'
    EMB_DIR = '/data/jx4237data/GNNteamingEvaluator/TeamingEvaluator/data_examine/SPECTER2_yearly_author_venue_embeddings'
    fn_author = os.path.join(EMB_DIR, f'author_embedding_{up_to_year}.json')
    if not os.path.isfile(fn_author):
        fn_author += '.gz'
    if os.path.isfile(fn_author):
        with (gzip.open(fn_author, 'rt') if fn_author.endswith('.gz') else open(fn_author)) as f:
            for aid, vec in json.load(f).items():
                if aid in AUTH_LIDX_OF:
                    _copy(AUTH_LIDX_OF[aid], vec, x_author)
        print(f'[dataset_builder]   ⇒ loaded {len(AUTH_LIDX_OF)} author embeddings from {fn_author}')

    fn_venue = os.path.join(EMB_DIR, f'venue_embedding_{up_to_year}.json')
    if not os.path.isfile(fn_venue):
        fn_venue += '.gz'
    if os.path.isfile(fn_venue):
        with (gzip.open(fn_venue, 'rt') if fn_venue.endswith('.gz') else open(fn_venue)) as f:
            for ven, vec in json.load(f).items():
                if ven in VEN_LIDX_OF:
                    _copy(VEN_LIDX_OF[ven], vec, x_venue)

    data['author'].x = x_author
    data['venue' ].x = x_venue

    # --------------------------------------------------------------- #
    # 3) edges (use the local contiguous indices)                     #
    # --------------------------------------------------------------- #
    # author → paper
    a_src, a_dst = [], []
    for pid in paper_ids:
        p = PAPER_LIDX_OF[pid]
        for aid in paper_json[pid]['neighbors']['author']:
            a_src.append(AUTH_LIDX_OF[aid]);  a_dst.append(p)
    data['author', 'writes', 'paper'].edge_index   = torch.tensor([a_src, a_dst])
    data['paper',  'written_by', 'author'].edge_index = torch.tensor([a_dst, a_src])

    # paper → paper
    p_src, p_dst = [], []
    for pid in paper_ids:
        p = PAPER_LIDX_OF[pid]
        for ref in paper_json[pid]['neighbors']['reference_papers']:
            if ref in PAPER_LIDX_OF and paper_json[ref]['features']['PubYear'] <= up_to_year:
                p_src.append(p);  p_dst.append(PAPER_LIDX_OF[ref])
    data['paper', 'cites',   'paper'].edge_index   = torch.tensor([p_src, p_dst])
    data['paper', 'cited_by','paper'].edge_index   = torch.tensor([p_dst, p_src])

    # paper → venue
    v_src, v_dst = [], []
    for pid in paper_ids:
        p = PAPER_LIDX_OF[pid]
        v = VEN_LIDX_OF[paper_json[pid]['features']['Venue']]
        v_src.append(p);  v_dst.append(v)
    data['paper', 'published_in', 'venue'].edge_index = torch.tensor([v_src, v_dst])
    data['venue', 'publishes',    'paper'].edge_index = torch.tensor([v_dst, v_src])

    # --------------------------------------------------------------- #
    # 4) store the raw-id lists inside the snapshot (needed later)    #
    # --------------------------------------------------------------- #
    data['paper'].raw_ids  = ACTIVE_PAPERS.copy()
    data['author'].raw_ids = ACTIVE_AUTHS.copy()
    data['venue'].raw_ids  = ACTIVE_VENUES.copy()

    return data