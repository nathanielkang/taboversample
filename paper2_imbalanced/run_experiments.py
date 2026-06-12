"""
run_experiments.py - Main experiment runner for Paper 2: TabOversample.

Orchestrates the full evaluation pipeline:
    For each dataset × oversampling method × random seed:
        1. Load dataset.
        2. Compute relevance function on y_train.
        3. Apply oversampling to produce (X_aug, y_aug).
        4. Train downstream regressors (CatBoost, XGBoost, MLP).
        5. Evaluate on the test set with all metrics.
        6. Record results.

After all experiments:
    - Print a summary table (method × dataset × metric).
    - Save raw results to CSV.
    - Save a LaTeX table ready to paste into the paper.

Usage
-----
    python run_experiments.py                          # full benchmark
    python run_experiments.py --dataset abalone        # single dataset
    python run_experiments.py --method tabover smoter   # subset of methods
    python run_experiments.py --seeds 5 --epochs 200   # custom settings
"""

import argparse
import logging
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from datasets import get_dataset, get_all_datasets, _DATASET_LOADERS
from relevance import relevance_function
from metrics import evaluate_all
from tabover import TabOversample
from baselines import (
    NoOversampling,
    RandomOversampler,
    SMOTEROversampler,
    SMOGNOversampler,
    VanillaTabDDPM,
    CTGANOversampler,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_experiments")


# ---------------------------------------------------------------------------
# Downstream regressors
# ---------------------------------------------------------------------------

def _get_regressors():
    """
    Return a dict  {name: constructor}  of downstream regressors.

    Each constructor takes no arguments and returns an sklearn-compatible
    estimator with .fit(X, y) and .predict(X).
    """
    regressors = {}

    # --- CatBoost ---
    try:
        from catboost import CatBoostRegressor
        regressors["CatBoost"] = lambda: CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            verbose=0, random_seed=42, thread_count=-1,
        )
    except ImportError:
        log.warning("catboost not installed - skipping CatBoost regressor. "
                    "Install via: pip install catboost")

    # --- XGBoost ---
    try:
        from xgboost import XGBRegressor
        regressors["XGBoost"] = lambda: XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            verbosity=0, random_state=42, n_jobs=-1,
        )
    except ImportError:
        log.warning("xgboost not installed - skipping XGBoost regressor. "
                    "Install via: pip install xgboost")

    # --- MLP ---
    from sklearn.neural_network import MLPRegressor
    regressors["MLP"] = lambda: MLPRegressor(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    if not regressors:
        raise RuntimeError("No downstream regressors available!")

    return regressors


# ---------------------------------------------------------------------------
# Oversampling method registry
# ---------------------------------------------------------------------------

def _get_methods(epochs: int, verbose: bool):
    """
    Return an ordered dict  {name: oversampler_instance}.
    """
    methods = {
        "None": NoOversampling(),
        "RandomOS": RandomOversampler(),
        "SMOTER": SMOTEROversampler(),
        "SMOGN": SMOGNOversampler(),
        "TabDDPM": VanillaTabDDPM(
            epochs=epochs, n_timesteps=1000, hidden_dim=256, n_layers=3,
            verbose=verbose,
        ),
        "CTGAN": CTGANOversampler(epochs=epochs, verbose=verbose),
        "TabOversample": None,  # handled separately (needs special API)
    }
    return methods


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single(dataset: dict, method_name: str, method_obj,
               regressor_name: str, regressor_ctor,
               seed: int, epochs: int, verbose: bool) -> dict:
    """
    Run one experiment:  (dataset, method, regressor, seed).

    Returns a dict of results.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train = dataset["X_train"].copy()
    y_train = dataset["y_train"].copy()
    X_test = dataset["X_test"].copy()
    y_test = dataset["y_test"].copy()

    # --- Oversampling ---
    t0 = time.time()

    if method_name == "TabOversample":
        model = TabOversample(
            input_dim=X_train.shape[1],
            hidden_dim=256, n_layers=3, n_timesteps=1000,
        )
        X_aug, y_aug = model.oversample(
            X_train, y_train,
            epochs=epochs, batch_size=256, lr=1e-3,
            verbose=verbose,
        )
    else:
        X_aug, y_aug = method_obj.oversample(X_train, y_train)

    oversample_time = time.time() - t0

    # --- Train downstream regressor ---
    t1 = time.time()
    reg = regressor_ctor()
    reg.fit(X_aug, y_aug)
    train_time = time.time() - t1

    # --- Predict & evaluate ---
    y_pred = reg.predict(X_test)

    phi_test = relevance_function(y_test, method="boxplot")
    result = evaluate_all(y_test, y_pred, phi_test, threshold=0.5)

    result.update({
        "dataset": dataset["name"],
        "method": method_name,
        "regressor": regressor_name,
        "seed": seed,
        "n_train_original": len(y_train),
        "n_train_augmented": len(y_aug),
        "oversample_sec": round(oversample_time, 1),
        "train_sec": round(train_time, 1),
    })

    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiments(args):
    """Execute the full experiment grid and save results."""

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Datasets ---
    if args.dataset == ["all"]:
        dataset_keys = list(_DATASET_LOADERS.keys())
    else:
        dataset_keys = args.dataset

    # --- Methods ---
    all_methods = _get_methods(epochs=args.epochs, verbose=args.verbose)
    if args.method == ["all"]:
        method_keys = list(all_methods.keys())
    else:
        # Map user-friendly names to registry keys
        name_map = {k.lower().replace("_", ""): k for k in all_methods}
        method_keys = []
        for m in args.method:
            key = m.lower().replace("_", "")
            if key in name_map:
                method_keys.append(name_map[key])
            else:
                log.warning(f"Unknown method '{m}' - skipping. "
                            f"Available: {list(all_methods.keys())}")

    # --- Regressors ---
    regressors = _get_regressors()

    # --- Experiment loop ---
    results = []
    total = (len(dataset_keys) * len(method_keys)
             * len(regressors) * args.seeds)
    log.info(f"Running {total} experiments  "
             f"({len(dataset_keys)} datasets × {len(method_keys)} methods × "
             f"{len(regressors)} regressors × {args.seeds} seeds)")

    exp_count = 0
    for ds_key in dataset_keys:
        try:
            dataset = get_dataset(ds_key)
        except Exception as e:
            log.error(f"Failed to load dataset '{ds_key}': {e}")
            continue

        for method_name in method_keys:
            method_obj = all_methods[method_name]

            for reg_name, reg_ctor in regressors.items():
                for seed in range(args.seeds):
                    exp_count += 1
                    tag = (f"[{exp_count}/{total}] "
                           f"{dataset['name']} | {method_name} | "
                           f"{reg_name} | seed={seed}")
                    log.info(tag)

                    try:
                        res = run_single(
                            dataset, method_name, method_obj,
                            reg_name, reg_ctor, seed=seed,
                            epochs=args.epochs, verbose=args.verbose,
                        )
                        results.append(res)
                        log.info(f"  -> RMSE={res['RMSE']:.4f}  "
                                 f"RMSE_rare={res['RMSE_rare']:.4f}  "
                                 f"SERA={res['SERA']:.4f}")
                    except Exception as e:
                        log.error(f"  FAILED: {e}")
                        results.append({
                            "dataset": dataset["name"],
                            "method": method_name,
                            "regressor": reg_name,
                            "seed": seed,
                            "error": str(e),
                        })

    # --- Aggregate & save ---
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "results_raw.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"Raw results saved to {csv_path}")

    # --- Summary table ---
    print_summary(df, output_dir)

    return df


# ---------------------------------------------------------------------------
# Pretty-print summary tables
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, output_dir: str):
    """
    Print and save summary tables:  mean ± std  over seeds,
    grouped by (method, regressor, dataset).
    """
    try:
        from tabulate import tabulate
    except ImportError:
        log.warning("tabulate not installed - using plain print. "
                    "Install via: pip install tabulate")
        tabulate = None

    metrics_of_interest = ["RMSE", "MAE", "RMSE_rare", "MAE_rare", "SERA"]

    # Filter out error rows
    df_ok = df.dropna(subset=["RMSE"]) if "RMSE" in df.columns else df

    if df_ok.empty:
        log.warning("No successful results to summarise.")
        return

    # --- Per-regressor summary ---
    for reg_name in df_ok["regressor"].unique():
        df_reg = df_ok[df_ok["regressor"] == reg_name]

        # Aggregate: mean ± std over seeds
        agg = (
            df_reg
            .groupby(["method", "dataset"])
            [metrics_of_interest]
            .agg(["mean", "std"])
        )

        # Flatten multi-level columns
        rows = []
        for (method, dataset), row in agg.iterrows():
            entry = {"Method": method, "Dataset": dataset}
            for m in metrics_of_interest:
                mu = row[(m, "mean")]
                sd = row[(m, "std")]
                if np.isnan(mu):
                    entry[m] = "N/A"
                else:
                    entry[m] = f"{mu:.4f}±{sd:.4f}"
            rows.append(entry)

        summary_df = pd.DataFrame(rows)

        print(f"\n{'='*80}")
        print(f"  RESULTS - Regressor: {reg_name}")
        print(f"{'='*80}")

        if tabulate is not None:
            print(tabulate(summary_df, headers="keys", tablefmt="grid",
                           showindex=False))
        else:
            print(summary_df.to_string(index=False))

        # Save CSV
        path = os.path.join(output_dir, f"summary_{reg_name}.csv")
        summary_df.to_csv(path, index=False)

    # --- LaTeX table ---
    _save_latex_table(df_ok, output_dir, metrics_of_interest)


def _save_latex_table(df: pd.DataFrame, output_dir: str,
                      metrics: list):
    """
    Generate a LaTeX table (one per regressor) ready for the paper.

    Format:  rows = methods,  columns = datasets × metric
    """
    for reg_name in df["regressor"].unique():
        df_reg = df[df["regressor"] == reg_name]

        agg = (
            df_reg
            .groupby(["method", "dataset"])
            [metrics]
            .agg(["mean", "std"])
        )

        datasets = df_reg["dataset"].unique().tolist()
        methods = df_reg["method"].unique().tolist()

        lines = []
        lines.append("% Auto-generated LaTeX table")
        lines.append(f"% Regressor: {reg_name}")
        lines.append(f"% Generated: {datetime.now().isoformat()}")
        lines.append("")

        # We produce one table per metric
        for metric in ["RMSE_rare", "SERA"]:
            n_ds = len(datasets)
            col_spec = "l" + "c" * n_ds
            lines.append(f"\\begin{{table}}[t]")
            lines.append(f"\\centering")
            lines.append(f"\\caption{{{metric} ({reg_name}) - "
                         f"mean $\\pm$ std over seeds.}}")
            lines.append(f"\\label{{tab:{metric.lower()}_{reg_name.lower()}}}")
            lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
            lines.append("\\toprule")
            header = "Method & " + " & ".join(datasets) + " \\\\"
            lines.append(header)
            lines.append("\\midrule")

            for method in methods:
                cells = [method.replace("_", "\\_")]
                for ds in datasets:
                    try:
                        mu = agg.loc[(method, ds), (metric, "mean")]
                        sd = agg.loc[(method, ds), (metric, "std")]
                        if np.isnan(mu):
                            cells.append("N/A")
                        else:
                            cells.append(f"{mu:.3f}$\\pm${sd:.3f}")
                    except KeyError:
                        cells.append("--")
                lines.append(" & ".join(cells) + " \\\\")

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}")
            lines.append("")

        tex_path = os.path.join(output_dir, f"table_{reg_name}.tex")
        with open(tex_path, "w") as f:
            f.write("\n".join(lines))
        log.info(f"LaTeX table saved to {tex_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="TabOversample Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", nargs="+", default=["all"],
        help="Dataset name(s) or 'all'. "
             f"Available: {list(_DATASET_LOADERS.keys())}",
    )
    parser.add_argument(
        "--method", nargs="+", default=["all"],
        help="Oversampling method(s) or 'all'. "
             "Available: None, RandomOS, SMOTER, SMOGN, TabDDPM, "
             "CTGAN, TabOversample",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of random seeds (default: 3).",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Diffusion / generative model training epochs (default: 100).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save results (default: results/).",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Show per-epoch progress bars for generative models.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 60)
    log.info("  TabOversample - Experiment Runner")
    log.info("=" * 60)
    log.info(f"  Datasets  : {args.dataset}")
    log.info(f"  Methods   : {args.method}")
    log.info(f"  Seeds     : {args.seeds}")
    log.info(f"  Epochs    : {args.epochs}")
    log.info(f"  Output    : {args.output_dir}")
    log.info("=" * 60)

    t_start = time.time()
    df = run_experiments(args)
    elapsed = time.time() - t_start

    log.info(f"\nAll experiments completed in {elapsed/60:.1f} minutes.")
    log.info(f"Results directory: {os.path.abspath(args.output_dir)}")
