#!/usr/bin/env python3
"""
Train / evaluate the simple GBM baseline (XGBoost).

Example
-------
python baseline_train.py --train_years 2006 2015 --test_years 2016 2019 --feature_type "author_only"
python baseline_train.py --train_years 2006 2015 --test_years 2016 2019 --feature_type "both" 
"""
import argparse, os, time, pickle
from rich.console import Console
from utils.data_utils import load_snapshots
from utils.metrics import male_vec, rmsle_vec, mape_vec
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns               # nicer default style, optional
sns.set()
from baselines.gbm_model import GBMBaseline
from scipy import stats


console = Console()



# ---------------------------------------------------------------------------
import torch
import numpy as np
import random

# ---------------------------------------------------------------------------
import torch
import numpy as np
import random

def build_dataset(snapshots, year_indices, years_all, feature_type="both", 
                 author_drop_mode=None, drop_k=1, random_seed=None):
    """
    Generates one (X, y) pair for every paper *published in the calendar year*
    designated by `year_indices`.

    Parameters:
    -----------
    snapshots : list of PyG Data objects
    year_indices : list of indices into snapshots
    years_all : list of calendar years corresponding to snapshots
    feature_type : str, one of {"both", "topic_only", "author_only"}
        - "both": include topic embeddings + author embeddings + author count (1537-dim)
        - "topic_only": include only topic embeddings (768-dim)
        - "author_only": include only author embeddings + author count (769-dim)
    author_drop_mode : str or None, one of {None, "first", "last", "random", 
                                       "keep_first", "keep_last", "drop_first_last"}
        - None: no author dropping (default behavior)
        - "first": drop first author embedding (replace with zeros)
        - "last": drop last author embedding (replace with zeros)  
        - "random": randomly drop k authors (replace with zeros)
        - "keep_first": keep only first author, drop all others
        - "keep_last": keep only last author, drop all others
        - "drop_first_last": drop both first and last authors, keep middle authors
    drop_k : int, number of authors to drop when author_drop_mode="random"
        If drop_k >= total authors, all authors are dropped (equivalent to topic_only)
    random_seed : int or None, seed for reproducible random author dropping

    Returns:
    --------
    X : [N, D] where D depends on feature_type
    y : [N, 5] - 5 yearly citation counts
    """
    if feature_type not in {"both", "topic_only", "author_only"}:
        raise ValueError("feature_type must be one of: 'both', 'topic_only', 'author_only'")
    
    valid_modes = {None, "first", "last", "random", "keep_first", "keep_last", "drop_first_last"}
    if author_drop_mode not in valid_modes:
        raise ValueError(f"author_drop_mode must be one of: {valid_modes}")
    
    if author_drop_mode is not None and feature_type == "topic_only":
        print("Warning: author_drop_mode has no effect when feature_type='topic_only'")
    
    # Set random seed for reproducible dropping
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
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
        y_citations  = g_now["paper"].y_citations [paper_ids].cpu().float() # [P,5]

        # Get topic embeddings if needed
        if feature_type in {"both", "topic_only"}:
            topic_now = g_now["paper"].x_title_emb[paper_ids].cpu()     # [P,768]

        # Get author information if needed
        if feature_type in {"both", "author_only"}:
            # author→paper incidence in the current year
            src, dst = g_now["author", "writes", "paper"].edge_index.cpu()

            # build a list of author indices for every *selected* paper
            pid_to_local = {int(pid): i for i, pid in enumerate(paper_ids)}
            paper2authors = [[] for _ in range(len(paper_ids))]
            for a, p in zip(src.tolist(), dst.tolist()):
                if p in pid_to_local:                         # ignore other years' papers
                    paper2authors[pid_to_local[p]].append(a)

            # author embeddings from the previous year (safe – no look-ahead)
            auth_emb_prev = (g_prev["author"].x.cpu()
                             if g_prev is not None and "author" in g_prev.node_types
                             else None)

        # --------------------------- build feature rows -------------------
        for idx_local, pid in enumerate(paper_ids):
            feature_components = []

            # Add topic features
            if feature_type in {"both", "topic_only"}:
                topic_vec = topic_now[idx_local]
                feature_components.append(topic_vec)

            # Add author features
            if feature_type in {"both", "author_only"}:
                auth_ids = paper2authors[idx_local]
                original_auth_count = len(auth_ids)
                
                # Apply author dropping if specified
                if author_drop_mode is not None and auth_ids:
                    auth_ids_to_use = auth_ids.copy()
                    
                    if author_drop_mode == "first":
                        # Drop first author (index 0)
                        auth_ids_to_use = auth_ids_to_use[1:]
                        
                    elif author_drop_mode == "last":
                        # Drop last author (index -1)
                        auth_ids_to_use = auth_ids_to_use[:-1]
                        
                    elif author_drop_mode == "keep_first":
                        # Keep only first author
                        auth_ids_to_use = auth_ids_to_use[:1]
                        
                    elif author_drop_mode == "keep_last":
                        # Keep only last author
                        auth_ids_to_use = auth_ids_to_use[-1:]
                        
                    elif author_drop_mode == "drop_first_last":
                        # Drop both first and last, keep middle authors
                        if len(auth_ids_to_use) <= 2:
                            # If only 1-2 authors, drop all
                            auth_ids_to_use = []
                        else:
                            # Keep middle authors (exclude first and last)
                            auth_ids_to_use = auth_ids_to_use[1:-1]
                        
                    elif author_drop_mode == "random":
                        # Randomly drop k authors
                        if drop_k >= len(auth_ids_to_use):
                            # Drop all authors
                            auth_ids_to_use = []
                        else:
                            # Randomly select authors to keep
                            keep_count = len(auth_ids_to_use) - drop_k
                            auth_ids_to_use = random.sample(auth_ids_to_use, keep_count)
                else:
                    auth_ids_to_use = auth_ids
                
                # Calculate mean author embedding from remaining authors
                if auth_ids_to_use and auth_emb_prev is not None:
                    valid_ids = [aid for aid in auth_ids_to_use
                                 if aid < auth_emb_prev.size(0)]
                    mean_auth = (auth_emb_prev[valid_ids].mean(0)
                                 if valid_ids else torch.zeros(768))
                else:
                    mean_auth = torch.zeros(768)
                
                # Number of authors (use original count, not after dropping)
                # This preserves information about the paper's original author count
                num_auth = torch.tensor([original_auth_count], dtype=torch.float32)
                
                feature_components.extend([mean_auth, num_auth])

            # Concatenate all feature components
            x = torch.cat(feature_components)

            feats .append(x.numpy())
            labels.append(y_citations[idx_local].numpy())

    X = np.asarray(feats,  dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    
    # Print feature dimensionality and dropping info
    if len(X) > 0:
        feature_dims = {
            "both": 1537,      # 768 + 768 + 1
            "topic_only": 768,  # 768
            "author_only": 769  # 768 + 1
        }
        expected_dim = feature_dims[feature_type]
        actual_dim = X.shape[1]
        
        drop_info = ""
        if author_drop_mode is not None:
            if author_drop_mode == "random":
                drop_info = f" (dropping {drop_k} authors randomly)"
            else:
                drop_info = f" (dropping {author_drop_mode} author)"
        
        print(f"Feature type: {feature_type}{drop_info}")
        print(f"Expected dim: {expected_dim}, Actual dim: {actual_dim}, Samples: {len(X)}")
        
    return X, y

def evaluate(model, X, y_true):
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0.0, None)      # ← keep ≥0

    y_true_t = torch.from_numpy(y_true)
    y_pred_t = torch.from_numpy(y_pred)

    male  = male_vec (y_true_t, y_pred_t)
    rmsle = rmsle_vec(y_true_t, y_pred_t)
    mape  = mape_vec (y_true_t, y_pred_t)
    return male, rmsle, mape, y_pred


# ───────────────────────────────────────────────────────────────────
# plotting helper  (now with save_path)
# ───────────────────────────────────────────────────────────────────
def plot_pred_true_distributions_with_ci(y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        horizons=None,
                                        bins: int = 2,
                                        confidence_level: float = 0.95,
                                        n_bootstrap: int = 100,
                                        save_path: str | None = None,
                                        show: bool = True):
    """
    Creates 1×L panel of KDE plots comparing prediction vs. truth with confidence intervals.

    Parameters
    ----------
    y_true, y_pred : array shape (N, L)
    horizons       : list/tuple of length L with axis titles
    bins           : histogram bin count (not used with KDE but kept for compatibility)
    confidence_level : confidence level for intervals (default 0.95)
    n_bootstrap    : number of bootstrap samples for confidence intervals
    save_path      : if given, the figure is written to that file
    show           : call plt.show(); set False if running head-less
    """
    if horizons is None:
        horizons = [f"Year {i} "  for i in range(y_true.shape[1])]

    L = len(horizons)
    fig, axes = plt.subplots(1, L, figsize=(4 * L, 3), sharey=True)
    
    # Ensure axes is always a list for consistent indexing
    if L == 1:
        axes = [axes]

    alpha = 1 - confidence_level
    
    for h in range(L):
        ax = axes[h]
        
        # Plot main KDE
        sns.kdeplot(y_true[:, h], ax=ax, label="True", color='tab:blue')
        sns.kdeplot(y_pred[:, h], ax=ax, label="Pred", color='tab:orange')
        
        # Add confidence intervals using bootstrap
        x_range = np.linspace(0, 20, 200)
        
        # Bootstrap for true data
        true_densities = []
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = np.random.choice(y_true[:, h], size=len(y_true[:, h]), replace=True)
            kde = stats.gaussian_kde(bootstrap_sample)
            true_densities.append(kde(x_range))
        
        true_densities = np.array(true_densities)
        true_lower = np.percentile(true_densities, 100 * alpha/2, axis=0)
        true_upper = np.percentile(true_densities, 100 * (1 - alpha/2), axis=0)
        
        # Bootstrap for predicted data
        pred_densities = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(y_pred[:, h], size=len(y_pred[:, h]), replace=True)
            kde = stats.gaussian_kde(bootstrap_sample)
            pred_densities.append(kde(x_range))
        
        pred_densities = np.array(pred_densities)
        pred_lower = np.percentile(pred_densities, 100 * alpha/2, axis=0)
        pred_upper = np.percentile(pred_densities, 100 * (1 - alpha/2), axis=0)
        
        # Plot confidence intervals
        ax.fill_between(x_range, true_lower, true_upper, alpha=0.3, color='tab:blue', 
                       label=f"True {confidence_level:.0%} CI")
        ax.fill_between(x_range, pred_lower, pred_upper, alpha=0.3, color='tab:orange',
                       label=f"Pred {confidence_level:.0%} CI")
        
        ax.set_title(horizons[h])
        ax.set_xlabel("Citation count in year")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 0.35)
        ax.legend()

    fig.suptitle("Predicted vs. True citation-count distributions with Confidence Intervals", fontsize=16)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure written to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig






# ───────────────────────────────────────────────────────────────────
# HIGH-QUALITY YEAR-WISE AGGREGATE PLOT
# ───────────────────────────────────────────────────────────────────
def plot_yearly_aggregates(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           horizons=None,
                           agg_fn=np.median,
                           show_iqr: bool = True,
                           save_path: str | None = None,
                           show: bool = True):
    """
    One panel, x-axis = forecast horizon, y-axis = agg_fn(citations).
    Draws True vs Pred with optional shaded IQR bands.
    """
    if horizons is None:
        horizons = [f"Year {i}" for i in range(y_true.shape[1])]

    L  = len(horizons)
    x  = np.arange(L)
    dx = 0.0                       # horizontal offset to avoid overlap

    # Aggregates
    agg_true = agg_fn(y_true, axis=0)
    agg_pred = agg_fn(y_pred, axis=0)

    # Spread (for IQR shading)
    if show_iqr:
        q25_true, q75_true = np.percentile(y_true, [25, 75], axis=0)
        q25_pred, q75_pred = np.percentile(y_pred, [25, 75], axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))

    # --- TRUE line -------------------------------------------------
    ax.plot(x - dx, agg_true, marker='o', ms=5, lw=1.5, color='tab:blue',
            label='True')
    if show_iqr:
        ax.fill_between(x - dx, q25_true, q75_true,
                        color='tab:blue', alpha=0.20)

    # --- PRED line -------------------------------------------------
    ax.plot(x + dx, agg_pred, marker='s', ms=5, lw=1.5, ls='--',
            color='tab:orange', label='Pred')
    if show_iqr:
        ax.fill_between(x + dx, q25_pred, q75_pred,
                        color='tab:orange', alpha=0.20)

    # Cosmetics -----------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_xlabel("Forecast year")
    ax.set_ylabel(f"{agg_fn.__name__.capitalize()} citation count")
    ax.set_title("Year-wise aggregated citation counts (GBM)", fontsize=16)
    ax.legend(frameon=False)
    sns.despine(ax=ax)             # drop top/right spines
    ax.grid(alpha=.25, axis='y')

    fig.tight_layout()

    # Save / show ---------------------------------------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure written to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_years", nargs=2, type=int, required=True)
    parser.add_argument("--test_years",  nargs=2, type=int, required=True)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth",    type=int, default=6)
    parser.add_argument("--learning_rate",type=float, default=0.05)
    parser.add_argument("--device",       default="cpu")      # GBM is CPU anyway
    parser.add_argument("--feature_type", default="both",
                        choices=["both", "topic_only", "author_only"],
                        help="Feature type to use for training")
    parser.add_argument("--author_drop_mode", type=str, default=None,
                        help="Mode for dropping authors (if any)")
    parser.add_argument("--drop_k", type=int, default=1,
                        help="Number of authors to drop when author_drop_mode='random'")
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
    snapshots = load_snapshots("data/yearly_snapshots_specter2/G_{}.pt", years_all)

    # 2) index helpers -----------------------------------------------------
    year2idx = {y: i for i, y in enumerate(years_all)}

    train_indices = [year2idx[y] for y in
                     range(args.train_years[0], args.train_years[1] + 1)]
    test_indices  = [year2idx[y] for y in
                     range(args.test_years[0] , args.test_years[1]  + 1)]

    # 3) dataset -----------------------------------------------------------
    console.print("Building training set …")
    X_train, y_train = build_dataset(snapshots, train_indices, years_all, feature_type=args.feature_type, author_drop_mode=args.author_drop_mode, drop_k=args.drop_k) 
    console.print("Building   test   set …")
    X_test , y_test  = build_dataset(snapshots, test_indices , years_all, feature_type=args.feature_type, author_drop_mode=args.author_drop_mode, drop_k=args.drop_k)


    console.print(f"Train samples: {X_train.shape[0]:,}")
    console.print(f"Test  samples: {X_test.shape [0]:,}")

    # 4) model -------------------------------------------------------------
    # model = GBMBaseline(
    #     n_estimators = args.n_estimators,
    #     max_depth    = args.max_depth,
    #     learning_rate= args.learning_rate,
    # )
    # console.print("[cyan]Training GBM …[/cyan]")
    # model.fit(X_train, y_train)
    # load the model
    
    model = pickle.load(open('runs_gbm/20250522_200518/gbm_model.pkl', "rb"))

    # 5) evaluation --------------------------------------------------------
    male, rmsle, mape, y_prod = evaluate(model, X_test, y_test)
    console.print(f"[green]MALE : {male.tolist()}[/green]")
    console.print(f"[green]RMSLE: {rmsle.tolist()}[/green]")
    console.print(f"[green]MAPE : {mape.tolist()}[/green]")
    plot_pred_true_distributions_with_ci(
            y_test,
            y_prod,
            horizons=[f"Year {i}" for i in range(5)],
            save_path=f"figs/pred_vs_true_gbm_{args.author_drop_mode}.png",
            show=False)

    # 6) persist -----------------------------------------------------------
    # run_dir = os.path.join("runs_gbm",
    #                        time.strftime("%Y%m%d_%H%M%S"))
    # os.makedirs(run_dir, exist_ok=True)
    # with open(os.path.join(run_dir, "gbm_model.pkl"), "wb") as fh:
    #     pickle.dump(model, fh)
    # console.print(f"Model saved to {run_dir}/gbm_model.pkl")



    plot_yearly_aggregates(
            y_test,
            y_prod,
            horizons=[f"Year {i}" for i in range(5)],
            agg_fn=np.median,            # or np.mean
            show_iqr=True,
            save_path=f"./figs/yearly_median_iqr_{args.author_drop_mode}.png",
            show=False)                  # set True if GUI backend available


if __name__ == "__main__":
    main()