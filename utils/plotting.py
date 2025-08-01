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
import torch

# ───────────────────────────────────────────────────────────────────
# plotting helper  (now with save_path)
# ───────────────────────────────────────────────────────────────────
def plot_pred_true_distributions_with_ci(y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        horizons=None,
                                        bins: int = 30,
                                        plot_type: str = "kde",
                                        confidence_level: float = 0.95,
                                        n_bootstrap: int = 100,
                                        save_path: str | None = None,
                                        title: str | None = None,
                                        show: bool = True):
    """
    Creates 1×L panel of KDE plots or histograms comparing prediction vs. truth with confidence intervals.
    Uses log-transformed x-axis: log1p(x)

    Parameters
    ----------
    y_true, y_pred : array shape (N, L)
    horizons       : list/tuple of length L with axis titles
    bins           : histogram bin count (used when plot_type="hist")
    plot_type      : "kde" for kernel density estimation or "hist" for histogram
    confidence_level : confidence level for intervals (default 0.95)
    n_bootstrap    : number of bootstrap samples for confidence intervals
    save_path      : if given, the figure is written to that file
    show           : call plt.show(); set False if running head-less
    """
    if horizons is None:
        horizons = [f"Year {i} "  for i in range(1, y_true.shape[1] + 1)]

    L = len(horizons)
    fig, axes = plt.subplots(1, L, figsize=(4 * L, 3), sharey=True)
    
    # Ensure axes is always a list for consistent indexing
    if L == 1:
        axes = [axes]

    alpha = 1 - confidence_level
    
    # Calculate average total five-year citation counts
    total_true_5yr = np.sum(y_true, axis=1)  # Sum across all years for each paper
    total_pred_5yr = np.sum(y_pred, axis=1)  # Sum across all years for each paper
    avg_total_true_5yr = np.mean(total_true_5yr)
    avg_total_pred_5yr = np.mean(total_pred_5yr)
    
    # Calculate median total five-year citation counts
    median_total_true_5yr = np.median(total_true_5yr)
    median_total_pred_5yr = np.median(total_pred_5yr)
    
    # Calculate log values
    log_avg_total_true_5yr = np.log1p(avg_total_true_5yr)
    log_avg_total_pred_5yr = np.log1p(avg_total_pred_5yr)
    log_median_total_true_5yr = np.log1p(median_total_true_5yr)
    log_median_total_pred_5yr = np.log1p(median_total_pred_5yr)
    
    # Print the results for easy copying
    print(f"\n=== Average Total Five-Year Citation Counts ===")
    print(f"GT_AVG_total_five_year_Citation_Counts: {avg_total_true_5yr:.4f}")
    print(f"GT_AVG_total_five_year_Citation_Counts (log): {log_avg_total_true_5yr:.4f}")
    print(f"Pred_AVG_total_five_year_Citation_Counts: {avg_total_pred_5yr:.4f}")
    print(f"Pred_AVG_total_five_year_Citation_Counts (log): {log_avg_total_pred_5yr:.4f}")
    print(f"\n=== Median Total Five-Year Citation Counts ===")
    print(f"GT_MEDIAN_total_five_year_Citation_Counts: {median_total_true_5yr:.4f}")
    print(f"GT_MEDIAN_total_five_year_Citation_Counts (log): {log_median_total_true_5yr:.4f}")
    print(f"Pred_MEDIAN_total_five_year_Citation_Counts: {median_total_pred_5yr:.4f}")
    print(f"Pred_MEDIAN_total_five_year_Citation_Counts (log): {log_median_total_pred_5yr:.4f}")
    print(f"===============================================\n")
    
    for h in range(L):
        ax = axes[h]
        
        # Transform data using log1p(x)
        y_true_transformed = torch.log1p(torch.tensor(y_true[:, h])).numpy()
        y_pred_transformed = torch.log1p(torch.tensor(y_pred[:, h])).numpy()
        
        if plot_type.lower() == "kde":
            # Plot main KDE on transformed data
            sns.kdeplot(y_true_transformed, ax=ax, label="True", color='tab:blue')
            sns.kdeplot(y_pred_transformed, ax=ax, label="Pred", color='tab:orange')
        elif plot_type.lower() == "hist":
            # Plot histograms on transformed data
            ax.hist(y_true_transformed, bins=bins, alpha=0.6, label="True", 
                   color='tab:blue', density=True, edgecolor='black', linewidth=0.5)
            ax.hist(y_pred_transformed, bins=bins, alpha=0.6, label="Pred", 
                   color='tab:orange', density=True, edgecolor='black', linewidth=0.5)
        else:
            raise ValueError("plot_type must be either 'kde' or 'hist'")
        
        # Add confidence intervals using bootstrap
        # Define x_range in transformed space
        max_val = max(np.max(y_true_transformed), np.max(y_pred_transformed))
        min_val = min(np.min(y_true_transformed), np.min(y_pred_transformed))
        x_range = np.linspace(min_val, max_val, 200)
        
        # Bootstrap for true data
        true_densities = []
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = np.random.choice(y_true_transformed, size=len(y_true_transformed), replace=True)
            if plot_type.lower() == "kde":
                kde = stats.gaussian_kde(bootstrap_sample)
                true_densities.append(kde(x_range))
            elif plot_type.lower() == "hist":
                # For histograms, compute density using numpy histogram
                hist_counts, hist_edges = np.histogram(bootstrap_sample, bins=bins, 
                                                     range=(min_val, max_val), density=True)
                # Interpolate histogram to x_range
                hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
                interpolated = np.interp(x_range, hist_centers, hist_counts)
                true_densities.append(interpolated)
        
        true_densities = np.array(true_densities)
        true_lower = np.percentile(true_densities, 100 * alpha/2, axis=0)
        true_upper = np.percentile(true_densities, 100 * (1 - alpha/2), axis=0)

        # Bootstrap for predicted data
        pred_densities = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(y_pred_transformed, size=len(y_pred_transformed), replace=True)
            if plot_type.lower() == "kde":
                kde = stats.gaussian_kde(bootstrap_sample)
                pred_densities.append(kde(x_range))
            elif plot_type.lower() == "hist":
                # For histograms, compute density using numpy histogram
                hist_counts, hist_edges = np.histogram(bootstrap_sample, bins=bins, 
                                                     range=(min_val, max_val), density=True)
                # Interpolate histogram to x_range
                hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
                interpolated = np.interp(x_range, hist_centers, hist_counts)
                pred_densities.append(interpolated)
        
        pred_densities = np.array(pred_densities)
        pred_lower = np.percentile(pred_densities, 100 * alpha/2, axis=0)
        pred_upper = np.percentile(pred_densities, 100 * (1 - alpha/2), axis=0)
        
        # Plot confidence intervals
        ax.fill_between(x_range, true_lower, true_upper, alpha=0.3, color='tab:blue', 
                       label=f"True {confidence_level:.0%} CI")
        ax.fill_between(x_range, pred_lower, pred_upper, alpha=0.3, color='tab:orange',
                       label=f"Pred {confidence_level:.0%} CI")
        
        exit()

        ax.set_title(horizons[h])
        ax.set_xlabel("log1p(Citation count)")
        ax.set_ylabel("Density")
        ax.legend()

    # Add annotations for average total five-year citation counts
    annotation_text = f"GT_AVG_total_five_year_Citation_Counts: {avg_total_true_5yr:.2f} (log: {log_avg_total_true_5yr:.2f})\n"
    annotation_text += f"Pred_AVG_total_five_year_Citation_Counts: {avg_total_pred_5yr:.2f} (log: {log_avg_total_pred_5yr:.2f})"
    
    # Position annotation in the top-left corner of the figure
    fig.text(0.02, 0.98, annotation_text, transform=fig.transFigure, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(f"{title} Predicted vs. True citation-count distributions with Confidence Intervals (Log-transformed)", fontsize=16)
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
                           title: str | None = None,
                           show: bool = True):
    """
    One panel, x-axis = forecast horizon, y-axis = agg_fn(citations).
    Draws True vs Pred with optional shaded IQR bands.
    """
    if horizons is None:
        horizons = [f"Year {i}" for i in range(1, y_true.shape[1] + 1)]

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

    # Calculate average total five-year citation counts
    total_true_5yr = np.sum(y_true, axis=1)  # Sum across all years for each paper
    total_pred_5yr = np.sum(y_pred, axis=1)  # Sum across all years for each paper
    avg_total_true_5yr = np.mean(total_true_5yr)
    avg_total_pred_5yr = np.mean(total_pred_5yr)
    
    # Calculate median total five-year citation counts
    median_total_true_5yr = np.median(total_true_5yr)
    median_total_pred_5yr = np.median(total_pred_5yr)
    
    # Calculate log values
    log_avg_total_true_5yr = np.log1p(avg_total_true_5yr)
    log_avg_total_pred_5yr = np.log1p(avg_total_pred_5yr)
    log_median_total_true_5yr = np.log1p(median_total_true_5yr)
    log_median_total_pred_5yr = np.log1p(median_total_pred_5yr)
    
    # Print the results for easy copying
    print(f"\n=== Average Total Five-Year Citation Counts ===")
    print(f"GT_AVG_total_five_year_Citation_Counts: {avg_total_true_5yr:.4f}")
    print(f"GT_AVG_total_five_year_Citation_Counts (log): {log_avg_total_true_5yr:.4f}")
    print(f"Pred_AVG_total_five_year_Citation_Counts: {avg_total_pred_5yr:.4f}")
    print(f"Pred_AVG_total_five_year_Citation_Counts (log): {log_avg_total_pred_5yr:.4f}")
    print(f"\n=== Median Total Five-Year Citation Counts ===")
    print(f"GT_MEDIAN_total_five_year_Citation_Counts: {median_total_true_5yr:.4f}")
    print(f"GT_MEDIAN_total_five_year_Citation_Counts (log): {log_median_total_true_5yr:.4f}")
    print(f"Pred_MEDIAN_total_five_year_Citation_Counts: {median_total_pred_5yr:.4f}")
    print(f"Pred_MEDIAN_total_five_year_Citation_Counts (log): {log_median_total_pred_5yr:.4f}")
    print(f"===============================================\n")

    # Cosmetics -----------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_xlabel("Forecast year")
    ax.set_ylabel(f"{agg_fn.__name__.capitalize()} citation count")
    ax.set_title(f"{title} Year-wise aggregated citation counts (Ours)", fontsize=16)
    ax.legend(frameon=False)
    sns.despine(ax=ax)             # drop top/right spines
    ax.grid(alpha=.25, axis='y')
    ax.set_ylim(0, 10)
    
    # Add annotations for average total five-year citation counts
    annotation_text = f"GT_AVG_total_five_year_Citation_Counts: {avg_total_true_5yr:.2f} (log: {log_avg_total_true_5yr:.2f})\n"
    annotation_text += f"Pred_AVG_total_five_year_Citation_Counts: {avg_total_pred_5yr:.2f} (log: {log_avg_total_pred_5yr:.2f})"
    
    # Position annotation in the top-left corner of the plot
    ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
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