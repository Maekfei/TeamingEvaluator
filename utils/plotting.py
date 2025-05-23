import matplotlib.pyplot as plt
import seaborn as sns               # nicer default style, optional
import numpy as np
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

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