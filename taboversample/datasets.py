"""
datasets.py - Load and preprocess imbalanced regression benchmark datasets.

Each dataset is auto-downloaded via sklearn/OpenML and returned as a dict with:
    X_train, y_train, X_test, y_test, name, n_features, cat_indices

All numerical features are standardized (fit on train only).
Categorical features are label-encoded.
Split: 70/30 train/test, random_state=42.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _label_encode_categoricals(df, cat_cols):
    """Label-encode categorical columns in-place and return encoders."""
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def _prepare_dataset(X_df, y_arr, name, cat_cols=None, random_state=42):
    """
    Standard preprocessing pipeline shared by every dataset loader.

    Parameters
    ----------
    X_df : pd.DataFrame
        Raw feature matrix (may contain categoricals).
    y_arr : np.ndarray
        Target vector.
    name : str
        Human-readable dataset name.
    cat_cols : list[str] or None
        Names of categorical columns in X_df.
    random_state : int
        Seed for train/test split.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_test, y_test, name, n_features, cat_indices
    """
    if cat_cols is None:
        cat_cols = []

    # Label-encode categoricals
    X_df, _ = _label_encode_categoricals(X_df, cat_cols)

    # Convert everything to float
    X = X_df.values.astype(np.float32)
    y = y_arr.astype(np.float32)

    # Record which column indices are categorical (after encoding)
    cat_indices = [list(X_df.columns).index(c) for c in cat_cols]

    # Train / test split (70 / 30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    # Standardize numerical (non-categorical) features
    num_indices = [i for i in range(X.shape[1]) if i not in cat_indices]
    scaler = StandardScaler()
    if num_indices:
        X_train[:, num_indices] = scaler.fit_transform(X_train[:, num_indices])
        X_test[:, num_indices] = scaler.transform(X_test[:, num_indices])

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "name": name,
        "n_features": X_train.shape[1],
        "cat_indices": cat_indices,
    }


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------

def load_abalone():
    """Abalone - predict number of rings (proxy for age)."""
    data = fetch_openml("abalone", version=1, as_frame=True, parser="auto")
    df = data.data.copy()
    target = data.target.astype(float).values

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    return _prepare_dataset(df, target, "Abalone", cat_cols=cat_cols)


def load_california_housing():
    """California Housing - predict median house value."""
    data = fetch_california_housing(as_frame=True)
    df = data.data.copy()
    target = data.target.values

    return _prepare_dataset(df, target, "CaliforniaHousing")


def load_bike_sharing():
    """Bike Sharing Demand - predict total rental count."""
    data = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True, parser="auto")
    df = data.data.copy()
    target = data.target.astype(float).values

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    return _prepare_dataset(df, target, "BikeSharing", cat_cols=cat_cols)


def load_cpu_activity():
    """CPU Activity - predict usr (user-mode CPU usage)."""
    data = fetch_openml("cpu_act", version=1, as_frame=True, parser="auto")
    df = data.data.copy()
    target = data.target.astype(float).values

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    return _prepare_dataset(df, target, "CPUActivity", cat_cols=cat_cols)


def load_insurance():
    """Insurance - predict medical charges."""
    data = fetch_openml("insurance", version=1, as_frame=True, parser="auto")
    df = data.data.copy()
    target = data.target.astype(float).values

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    return _prepare_dataset(df, target, "Insurance", cat_cols=cat_cols)


def load_house_16h():
    """House 16H - predict median house price."""
    data = fetch_openml("house_16H", version=1, as_frame=True, parser="auto")
    df = data.data.copy()
    target = data.target.astype(float).values

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    return _prepare_dataset(df, target, "House16H", cat_cols=cat_cols)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DATASET_LOADERS = {
    "abalone": load_abalone,
    "california_housing": load_california_housing,
    "bike_sharing": load_bike_sharing,
    "cpu_activity": load_cpu_activity,
    "insurance": load_insurance,
    "house_16h": load_house_16h,
}


def get_dataset(name: str) -> dict:
    """
    Load a single dataset by name.

    Parameters
    ----------
    name : str
        One of: abalone, california_housing, bike_sharing,
                cpu_activity, insurance, house_16h

    Returns
    -------
    dict  (X_train, y_train, X_test, y_test, name, n_features, cat_indices)
    """
    key = name.lower().replace(" ", "_")
    if key not in _DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(_DATASET_LOADERS.keys())}"
        )
    print(f"[datasets] Loading {key} ...")
    return _DATASET_LOADERS[key]()


def get_all_datasets() -> list:
    """Load and return all benchmark datasets as a list of dicts."""
    datasets = []
    for key in _DATASET_LOADERS:
        try:
            ds = get_dataset(key)
            print(f"  -> {ds['name']}: train={ds['X_train'].shape}, "
                  f"test={ds['X_test'].shape}, features={ds['n_features']}")
            datasets.append(ds)
        except Exception as e:
            print(f"  [WARN] Failed to load {key}: {e}")
    return datasets


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = get_all_datasets()
    print(f"\nLoaded {len(datasets)} datasets successfully.")
    for ds in datasets:
        print(f"  {ds['name']:25s}  train={ds['X_train'].shape}  "
              f"test={ds['X_test'].shape}  cat_idx={ds['cat_indices']}")
