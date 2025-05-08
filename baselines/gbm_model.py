"""
A very thin wrapper around XGBoost that mimics a minimal sklearn
interface  (fit → predict)  and handles multi-output regression.
"""
from sklearn.multioutput import MultiOutputRegressor
from xgboost            import XGBRegressor


class GBMBaseline:
    def __init__(self, **xgb_params):
        """
        Parameters are passed straight to XGBRegressor, e.g.
        n_estimators, max_depth, learning_rate, subsample, …
        """
        default = dict(objective="reg:squarederror",
                       n_jobs=8,
                       tree_method="hist",
                       random_state=42)
        default.update(xgb_params)

        self.model = MultiOutputRegressor(
            XGBRegressor(**default)
        )

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """
        X : ndarray [N, D]
        y : ndarray [N, 5]
        """
        self.model.fit(X, y)
        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        """
        Returns ndarray [N, 5]
        """
        return self.model.predict(X)