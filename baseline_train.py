#!/usr/bin/env python3
"""
Train / evaluate the simple GBM baseline (XGBoost).

Example
-------
python baseline_train.py --train_years 2006 2015 --test_years 2016 2019
"""
import argparse, os, time, pickle
import numpy as np
import torch
from rich.console import Console
from utils.data_utils import load_snapshots
from utils.metrics import male_vec, rmsle_vec

from baselines.gbm_model import GBMBaseline

console = Console()


# ---------------------------------------------------------------------------
def build_dataset(snapshots, year_indices, years_all):      # ← extra arg
    """
    Generates one (X, y) pair for every paper *published in the calendar year*
    designated by `year_indices`.

    X  : [N, 513]      256 topic emb + 256 mean-author emb + 1 (#authors)
    y  : [N, 5]        5 yearly citation counts
    """
    feats, labels = [], []

    for t in year_indices:
        g_now       = snapshots[t]
        g_prev      = snapshots[t - 1] if t - 1 >= 0 else None
        curr_year   = years_all[t]                      # ← true calendar year

        # ----------- pick only the papers published *this* year -----------
        mask = (g_now["paper"].y_year == curr_year).cpu()
        if mask.sum() == 0:                 # no new papers in this snapshot
            continue

        paper_ids    = mask.nonzero(as_tuple=False).view(-1)
        topic_now    = g_now["paper"].x_title_emb[paper_ids].cpu()     # [P,256]
        y_citations  = g_now["paper"].y_citations [paper_ids].cpu().float() # [P,5]

        # author→paper incidence in the current year
        src, dst = g_now["author", "writes", "paper"].edge_index.cpu()

        # build a list of author indices for every *selected* paper
        pid_to_local = {int(pid): i for i, pid in enumerate(paper_ids)}
        paper2authors = [[] for _ in range(len(paper_ids))]
        for a, p in zip(src.tolist(), dst.tolist()):
            if p in pid_to_local:                         # ignore other years’ papers
                paper2authors[pid_to_local[p]].append(a)

        # author embeddings from the previous year (safe – no look-ahead)
        auth_emb_prev = (g_prev["author"].x.cpu()
                         if g_prev is not None and "author" in g_prev.node_types
                         else None)

        # --------------------------- build feature rows -------------------
        for idx_local, pid in enumerate(paper_ids):
            topic_vec = topic_now[idx_local]

            auth_ids = paper2authors[idx_local]
            if auth_ids and auth_emb_prev is not None:
                valid_ids = [aid for aid in auth_ids
                             if aid < auth_emb_prev.size(0)]
                mean_auth = (auth_emb_prev[valid_ids].mean(0)
                             if valid_ids else torch.zeros(256))
            else:
                mean_auth = torch.zeros(256)

            num_auth = torch.tensor([len(auth_ids)], dtype=torch.float32)
            x = torch.cat([topic_vec, mean_auth, num_auth])   # 513-dim

            feats .append(x.numpy())
            labels.append(y_citations[idx_local].numpy())

    X = np.asarray(feats,  dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return X, y
# ---------------------------------------------------------------------------


def evaluate(model, X, y_true):
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0.0, None)      # ← keep ≥0

    y_true_t = torch.from_numpy(y_true)
    y_pred_t = torch.from_numpy(y_pred)

    male  = male_vec (y_true_t, y_pred_t)
    rmsle = rmsle_vec(y_true_t, y_pred_t)
    return male, rmsle


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_years", nargs=2, type=int, required=True)
    parser.add_argument("--test_years",  nargs=2, type=int, required=True)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth",    type=int, default=6)
    parser.add_argument("--learning_rate",type=float, default=0.05)
    parser.add_argument("--device",       default="cpu")      # GBM is CPU anyway
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # 1) load all snapshots we might need
    #    we also read *one year before* the first training year to obtain
    #    author embeddings for that very first year.
    # ----------------------------------------------------------------------
    first_needed = args.train_years[0] - 1
    years_all = list(range(first_needed,
                           args.test_years[1] + 1))
    console.print(f"Loading {len(years_all)} yearly graphs …")
    snapshots = load_snapshots("data/raw/G_{}.pt", years_all)

    # 2) index helpers -----------------------------------------------------
    year2idx = {y: i for i, y in enumerate(years_all)}

    train_indices = [year2idx[y] for y in
                     range(args.train_years[0], args.train_years[1] + 1)]
    test_indices  = [year2idx[y] for y in
                     range(args.test_years[0] , args.test_years[1]  + 1)]

    # 3) dataset -----------------------------------------------------------
    console.print("Building training set …")
    X_train, y_train = build_dataset(snapshots, train_indices, years_all)   # ← add years_all
    console.print("Building   test   set …")
    X_test , y_test  = build_dataset(snapshots, test_indices , years_all)   # ← add years_all


    console.print(f"Train samples: {X_train.shape[0]:,}")
    console.print(f"Test  samples: {X_test.shape [0]:,}")

    # 4) model -------------------------------------------------------------
    model = GBMBaseline(
        n_estimators = args.n_estimators,
        max_depth    = args.max_depth,
        learning_rate= args.learning_rate,
    )
    console.print("[cyan]Training GBM …[/cyan]")
    model.fit(X_train, y_train)

    # 5) evaluation --------------------------------------------------------
    male, rmsle = evaluate(model, X_test, y_test)
    console.print(f"[green]MALE : {male.tolist()}[/green]")
    console.print(f"[green]RMSLE: {rmsle.tolist()}[/green]")

    # 6) persist -----------------------------------------------------------
    run_dir = os.path.join("runs_gbm",
                           time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "gbm_model.pkl"), "wb") as fh:
        pickle.dump(model, fh)
    console.print(f"Model saved to {run_dir}/gbm_model.pkl")


if __name__ == "__main__":
    main()