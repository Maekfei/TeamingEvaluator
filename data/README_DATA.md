# Expected data objects

Each `data/raw/G_<YEAR>.pt` **must** be a pickled
`torch_geometric.data.HeteroData` that contains:

data.x_title_emb          # for 'paper' nodes, float32 [N_papers, d_title]
data['paper'].y_citations # int64   [N_papers, L]      (L yrs ahead)


Optional: pre-computed masks `train_mask`, `test_mask` etc.

See `utils/data_utils.py -> build_dummy_snapshot()` for a minimal example.