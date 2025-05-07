# baselines/gbm_model.py
from __future__ import annotations
import numpy as np
import xgboost as xgb
from utils.metrics import male_vec, rmsle_vec
import torch

class GBMBaseline:
    """
    Wrapper around XGBoost for five-step citation prediction.
    1) extract hand-crafted features per paper
    2) train five independent regressors (one per horizon)

    Feature engineering is deliberately left as TODO().
    """

    def __init__(self, device: str = "cpu"):
        self.models: list[xgb.XGBRegressor] | None = None
        self.device = device                        # for future GPU use

    # ------------------------------------------------------------------
    # Public API identical to the deep-learning model
    # ------------------------------------------------------------------
    def fit(self, snapshots, years_train):
        X_list, Y_list = [], []
        for t in years_train:
            data = snapshots[t]
            y = data["paper"].y_citations.numpy()   # [N, 5]
            X = self._extract_features(data)        # [N, F]
            X_list.append(X)
            Y_list.append(y)

        X_train = np.concatenate(X_list, 0)
        Y_train = np.concatenate(Y_list, 0)         # [num_papers, 5]

        # one regressor per horizon
        self.models = []
        for k in range(5):
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                objective="reg:squarederror",
            )
            model.fit(X_train, Y_train[:, k])
            self.models.append(model)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, snapshots, years_test):
        male_all, rmsle_all = [], []
        for t in years_test:
            data  = snapshots[t]
            y_true = data["paper"].y_citations.numpy()          # [N, 5]
            X      = self._extract_features(data)               # [N, F]

            # predict horizon-wise then stack
            preds = [m.predict(X) for m in self.models]         # list(5)[N]
            y_hat = np.stack(preds, 1)

            y_true_t = torch.tensor(y_true)
            y_hat_t  = torch.tensor(y_hat)
            male_all .append(male_vec (y_true_t, y_hat_t))
            rmsle_all.append(rmsle_vec(y_true_t, y_hat_t))

        return (torch.stack(male_all ).mean(0).cpu(),
                torch.stack(rmsle_all).mean(0).cpu())

    # ------------------------------------------------------------------
    # ---------- feature engineering placeholder -----------------------
    # ------------------------------------------------------------------
    def _extract_features(self, data):
        """
        Return numpy array [num_papers, F].

        Insert the hand-crafted features from
        Shi et al. (2021) & Shen et al. (2014) here.
        """
        N = data["paper"].num_nodes
        # -------- Example dummy feature: paper-length (replace!) --------
        feat_dummy = torch.ones(N, 1)
        # → add more features and concatenate
        feats = feat_dummy                        # Tensor [N, F]
        return feats.numpy()
