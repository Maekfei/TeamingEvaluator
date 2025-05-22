import matplotlib.pyplot as plt
import seaborn as sns               # nicer default style, optional
import numpy as np
import os
import warnings


# ───────────────────────────────────────────────────────────────────
# plotting helper  (now with save_path)
# ───────────────────────────────────────────────────────────────────
def plot_pred_true_distributions(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 horizons=None,
                                 bins: int = 2,
                                 save_path: str | None = None,
                                 show: bool = True):
    """
    Creates 1×L panel of histograms (or KDEs) comparing prediction vs. truth.

    Parameters
    ----------
    y_true, y_pred : array shape (N, L)
    horizons       : list/tuple of length L with axis titles
    bins           : histogram bin count
    save_path      : if given, the figure is written to that file
                     (e.g. './figs/citations.png')
    show           : call plt.show(); set False if running head-less
    """
    if horizons is None:
        horizons = [f"Year {i} "  for i in range(y_true.shape[1])]

    L = len(horizons)
    fig, axes = plt.subplots(1, L, figsize=(4 * L, 3), sharey=True)

    for h in range(L):
        ax = axes[h]
        # ax.hist(y_true[:, h], bins=bins, alpha=.50, label="True", color="tab:blue")
        # ax.hist(y_pred[:, h], bins=bins, alpha=.50, label="Pred", color="tab:orange")
        sns.kdeplot(y_true[:, h], ax=ax, label="True")
        sns.kdeplot(y_pred[:, h], ax=ax, label="Pred")
        ax.set_title(horizons[h])
        ax.set_xlabel("Citation count in year")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 0.35)
        # if h == 0:
        #     ax.set_ylabel("Frequency")
        ax.legend()
    

    fig.suptitle("Predicted vs. True citation-count distributions (Ours)", fontsize=16)
    fig.tight_layout()

    # --------------------------------------------------------------
    # save or show
    # --------------------------------------------------------------
    if save_path is not None:
        # create directory if it does not exist
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
    ax.set_title("Year-wise aggregated citation counts (Ours)", fontsize=16)
    ax.legend(frameon=False)
    sns.despine(ax=ax)             # drop top/right spines
    ax.grid(alpha=.25, axis='y')
    ax.set_ylim(0, 10)
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