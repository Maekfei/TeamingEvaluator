import torch


def male_vec(y_true, y_pred):
    """
    Mean Absolute Log-scaled Error for every horizon separately.
    Returns tensor [L]
    """
    err = torch.abs(torch.log1p(y_pred) - torch.log1p(y_true))
    return err.mean(dim=0)


def rmsle_vec(y_true, y_pred):
    """
    Root Mean Square Log-scaled Error  for every horizon.
    """
    err2 = (torch.log1p(y_pred) - torch.log1p(y_true)) ** 2
    return torch.sqrt(err2.mean(dim=0))


def mape_vec(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1.0):
    """
    Mean Absolute Percentage Error per horizon.
    • eps is added to the denominator so that divisions by small
      (or zero) citation counts do not explode.
    Returns: tensor of length L  (L = # horizons)
    """
    denom = torch.clamp(torch.abs(y_true), min=eps)        # |y_true|  (≥ eps)
    pct_err = torch.abs(y_pred - y_true) / denom
    return pct_err.mean(dim=0)