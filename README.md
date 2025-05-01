# Dynamic Citation-Impact Prediction

This repository implements a three-stage framework for predicting the future citation trajectory of a scientific paper directly from the temporal bibliographic graph.

## Framework Components

1. **Temporally aligned R-GCN**
   - Shared weights encode every yearly snapshot of the heterogeneous graph (papers – authors – venues – citations).

2. **Weighted embedding imputation**
   - Before the paper exists in the graph, its embedding is imputed from neighbours (authors, venue, referenced papers).
   - A learnable scalar weight per neighbour type controls the mixture.

3. **Citation time-series generator**
   - A GRU consumes the 5-year pre-publication embedding sequence and an MLP head outputs the parameters η, μ, σ of a log-normal survival curve C ^ (l)=α⋅(exp(η⋅Φ((lnl−μ)/σ))−1) from which yearly citation counts are obtained.
   - The training loss is L=L<sub>pred</sub> +β⋅L<sub>time−smooth</sub> where L<sub>time−smooth</sub> penalises sudden changes of node embeddings between consecutive years.

## 1. Installation

```bash
# 0) get the code
https://github.com/jiaweixu98/TeamingEvaluator.git
cd cite-impact-prediction

# 1) create isolated Python ≥3.9 env (here: venv)
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) install dependencies
# GPU (CUDA 11.8) – tested on RTX A6000
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 \
torchaudio==2.3.0 --extra-index-url https://download.pytorch.org/whl/cu118

# remaining python packages
pip install -r requirements.txt

# CPU-only: remove the +cu118 tags in the pip install torch ... command
```

## 2. Data

### 2.1 Raw sources

- `/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/papernodes_remove0/paper_nodes_GNN_yearly.json.gz` – 2M PubMed papers with neighbours and yearly citation counts
- `/data/jx4237data/Graph-CoT/Pipeline/2024_updated_data/tkg_embeddings_all_2024.npz` – 768-d SciBERT embeddings for all papers

### 2.2 Automatic snapshot generation

When a file `data/raw/G_<year>.pt` does not exist, the first call to `utils/data_utils.load_snapshots` will:
- load the compressed JSON & NPZ
- build a `torch_geometric.data.HeteroData` that contains all papers published ≤ year
- write the snapshot to `data/raw/G_<year>.pt` for future runs

This happens only once per year; thereafter the pre-generated .pt files are loaded instantly.

## 3. Quick start

### 3.1 Smoke-test (1 epoch, tiny split)

```bash
python train.py \
  --train_years 1995 1995 \
  --test_years 1996 1996 \
  --hidden_dim 32 --epochs 1 --device cuda:0
```

Output (abbreviated):
```
Train years: [1995]
Test years: [1996]
Epoch 001 L_pred:2.33 L_time:0.39 Loss:2.52
Eval MALE tensor([...]) RMSLE tensor([...])
```

No `dataset_builder loading ...` messages appear if G_1995.pt and G_1996.pt are already cached.

### 3.2 Minimal experiment (2 train + 2 test years)

```bash
python train.py \
  --train_years 1995 1996 \
  --test_years 1997 1998 \
  --hidden_dim 50 --epochs 30 --device cuda:0
```

GPU memory ≈ 8 GB, runtime ≈ 25 min on a single A6000.

## 4. Directory structure

```
cite-impact-prediction/
│
├── README.md                     ← you are here
├── requirements.txt
│
├── data/
│   ├── raw/                      ← yearly snapshots G_<year>.pt (auto-generated)
│   └── yearly_snapshots/         ← intermediate build artefacts
│
├── models/
│   ├── rgcn_encoder.py
│   ├── imputer.py
│   ├── impact_rnn.py
│   └── full_model.py
│
├── utils/
│   ├── dataset_builder.py        ← JSON/NPZ → HeteroData converter
│   ├── data_utils.py             ← caching & loading helper
│   └── metrics.py                ← MALE, RMSLE
│
└── train.py                      ← training / evaluation entry point
```