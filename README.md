# Scientific Team Performance Evaluator

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


@@ Training options  (train.py)
   --beta               temporal-smoothness weight
   --hidden_dim         size of node / RNN embeddings
   --cold_start_prob    float ∈ [0,1].  With this probability each training
                       paper is treated as a cold-start example, i.e. its
                       venue and reference neighbours are removed before the
                       imputer is called.  Authors + topic remain intact.
   --device             cuda:N  or  cpu


## 1. Installation

```bash
# 0) get the code
git clone https://github.com/jiaweixu98/TeamingEvaluator.git
cd TeamingEvaluator

# 1) create isolated Python ≥3.9 env (here: venv)
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) install dependencies

pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# remaining python packages
pip install -r requirements.txt
```

## 2. Data

### 2.1 Raw sources

- `data/raw/paper_nodes_GNN_yearly.json.gz` – 2M PubMed papers with neighbours and yearly citation counts
- `data/raw/tkg_embeddings_all_2024.npz` – 768-d SPECTER2 embeddings for all papers

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
  --hidden_dim 32 \
  --epochs 1 \
  --cold_start_prob 0.5 \
  --device cuda:0
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
  --train_years 2005 2015 \
  --test_years 2015 2019 \
  --hidden_dim 64 \
  --epochs 50 \
  --cold_start_prob 0.5 \
  --device cuda:0
```
parser.add_argument("--train_years", nargs=2, type=int, required=True,
                        help="e.g. 1995 2004 inclusive")
    parser.add_argument("--test_years", nargs=2, type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)  # unused in v1
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=.5) # regularization parameter (temporal smoothing regularizer of the temporal graph, make sure the same papers are not too different in the two consecutive years)
    parser.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cold_start_prob", type=float, default=0.3,
                    help="probability that a training paper is treated as "
                         "venue/reference-free (cold-start calibration)")
    parser.add_argument("--eval_mode", choices=["paper", "team"], default="paper",
                    help="'paper' = original evaluation  |  'team' = counter-factual")

## 4. Directory structure

```
TeamingEvaluator/
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

Todos:
author embedding: weighted aggreagated. (now random)
venue embedding: do an average. (now random)

there are so many papers, lead to a very large graph.

read all the snapshots into a list at a same time, can improve it.

currently only 9k cits are used in training, the sample size is too small, the model itself is very complex (millions of nodes).

Current impute, is simple averaging, we can consider add some weights to understand different importance of the specific neibghbors.

Replacing random author/venue vectors with weighted averages of paper embeddings is safe and likely beneficial.

Combine dimensionality reduction, half precision, shared storage and sampled mini-batches to curb memory usage.

Balance the small counter-factual set with dual-task learning, synthetic masking and stricter regularisation so the complex model does not overfit to only 9 k examples.