"""Pipeline factory: every learned transformation lives inside the pipeline,
so cross-validation refits it on each training fold (no leakage)."""
from __future__ import annotations

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .features import FeatureMaker


def make_pipeline(model, scaler="standard", k_best=None, **fm_kwargs) -> Pipeline:
    steps = [
        ("features", FeatureMaker(**fm_kwargs)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", {"standard": StandardScaler(), "robust": RobustScaler(), None: "passthrough"}[scaler]),
        ("select", SelectKBest(f_classif, k=k_best) if k_best else "passthrough"),
        ("model", model),
    ]
    return Pipeline(steps)


def scores(pipe, X):
    """Continuous score used for ranking (probability if available)."""
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return pipe.decision_function(X)
