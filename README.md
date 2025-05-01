# Dynamic Citation Impact Prediction

This repository implements the three–stage framework proposed in the prompt:

1. **Temporally-aligned R-GCN** – encodes each yearly bibliometric snapshot;
2. **Weighted embedding imputation** – fabricates a pre-publication trajectory
   for every new paper using its metadata neighbours;
3. **Citation time-series generator** – a GRU encodes the imputed trajectory,
   then an MLP estimates three parameters (η, μ, σ) that define a log-normal
   citation survival curve.

The loss combines prediction error with a temporal-smoothness regulariser.


## Installation
```bash
python -m venv .venv
source .venv/bin/activate

pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0 --extra-index-url https://download.pytorch.org/whl/cu118


If # no data/raw/*.pt on disk yet
python train.py --train_years 1995 1997 --test_years 1998 1999 --epochs 2


0) get code
git clone <this-repo-url> && cd cite-impact-prediction

1) create isolated python 3.10 environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

2) install deps (cpu; for cuda see README)
pip install -r requirements.txt

3) place yearly snapshot files into data/raw/
Each file G_<year>.pt must contain a torch_geometric.data.HeteroData
(see data/README_DATA.md for detailed schema)
4) train & test
python train.py 
--train_years 1995 2004 
--test_years  2005 2010 
--epochs 30 
--batch_size 256

Results (loss, MALE, RMSLE for each horizon l=1,2,5) are printed and also
written to runs/<timestamp>/.



Project structure
cite-impact-prediction/
│
├── README.md              ← Usage, data format, how the code works
├── requirements.txt
│
├── data/                  ← Put raw *.pt or *.pkl graphs here
│   └── README_DATA.md     ← Expected data format (one HeteroData per year)
│
├── models/
│   ├── rgcn_encoder.py    ← R-GCN that shares parameters across years
│   ├── imputer.py         ← Weighted embedding imputation module
│   ├── impact_rnn.py      ← GRU + MLP that outputs η, μ, σ
│   └── full_model.py      ← High-level wrapper, loss, metrics
│
├── utils/
│   ├── data_utils.py      ← Load snapshots, collate, negative sampling…
│   └── metrics.py         ← MALE & RMSLE
│
└── train.py               ← Everything tied together