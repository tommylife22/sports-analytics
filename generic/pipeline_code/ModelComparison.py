# ModelComparison.py
# Module for comparing model performance metrics

def compare_model_metrics(models):
    """
    Compare performance metrics across models, automatically detecting problem type.
    For classification, adds AUC (if binary) and can show ROC curves.

    Parameters
    ----------
    models : dict[str, object] | object
        Dict of name->model or a single model instance.

    Returns
    -------
    pandas.DataFrame
        Table of metrics by model.
    """
    import pandas as pd
    import numpy as np

    # Optional helpers (guarded imports)
    try:
        from .rocCalculation import plot_roc_curves, add_auc_to_models
    except Exception:
        plot_roc_curves = None
        add_auc_to_models = None

    # --- Normalize input to dict ---
    if not isinstance(models, dict):
        m = models
        name = getattr(m, "model_name", m.__class__.__name__)
        models = {name: m}

    # --- Determine problem type from a sample model (fallback to 'classification') ---
    sample_model = next(iter(models.values()))
    problem_type = getattr(sample_model, "problem_type", "classification")

    # Try to detect binary classification to decide on AUC/ROC
    y_test = None
    is_binary = False
    if problem_type == "classification":
        try:
            # split_data signature assumed on your wrapper class
            _, X_test, _, y_test = sample_model.split_data(random_state=42)
            # handle Series/array
            y_arr = np.array(y_test)
            uniq = np.unique(y_arr[~pd.isna(y_arr)])
            is_binary = (uniq.size == 2)
        except Exception:
            # if anything fails, leave is_binary False
            pass

    # --- Optionally add AUC to models (only if we can and helper exists) ---
    if problem_type == "classification" and is_binary and add_auc_to_models is not None:
        # n_runs per model may differ; add_auc_to_models handles dict-level
        # If your add_auc_to_models requires a unified n_runs, we try to infer a sane default.
        try:
            # Prefer sample model n_runs, else 1
            n_runs = getattr(sample_model, "n_runs", 1)
            models = add_auc_to_models(models, n_runs, include_train=True)
        except Exception:
            # Don't fail comparison if AUC calc fails
            pass

    # --- Collect metrics per model (defensive on attributes) ---
    rows = []
    all_columns = set(["Model"])
    minimize_metrics = {"mse", "rmse", "mae"}  # extend as needed

    # We'll keep track of discovered "Test ..." columns to pick best models later
    discovered_test_cols = set()

    for name, model in models.items():
        row = {"Model": name}

        # detect averaged-vs-single per model
        has_avg_test = hasattr(model, "avg_test_metrics") and getattr(model, "avg_test_metrics") is not None
        has_avg_train = hasattr(model, "avg_train_metrics") and getattr(model, "avg_train_metrics") is not None
        using_avg_for_this_model = bool(has_avg_test and has_avg_train)

        # pick metrics lists per model
        if using_avg_for_this_model:
            test_metrics = [k for k in model.avg_test_metrics.keys() if not k.endswith("_std")]
            train_metrics = [k for k in model.avg_train_metrics.keys() if not k.endswith("_std")]
            n_runs = getattr(model, "n_runs", getattr(sample_model, "n_runs", 1))
        else:
            test_metrics = list(getattr(model, "test_metrics", {}).keys())
            train_metrics = list(getattr(model, "train_metrics", {}).keys())
            n_runs = 1

        # For classification: ensure AUC appears if present
        if problem_type == "classification":
            # If AUC exists, make sure it is in the metric lists
            if using_avg_for_this_model:
                if "auc" in getattr(model, "avg_test_metrics", {}):
                    if "auc" not in test_metrics: test_metrics.append("auc")
                if "auc" in getattr(model, "avg_train_metrics", {}):
                    if "auc" not in train_metrics: train_metrics.append("auc")
            else:
                if "auc" in getattr(model, "test_metrics", {}):
                    if "auc" not in test_metrics: test_metrics.append("auc")
                if "auc" in getattr(model, "train_metrics", {}):
                    if "auc" not in train_metrics: train_metrics.append("auc")

        # Fill row columns
        if using_avg_for_this_model:
            for metric in train_metrics:
                if metric in model.avg_train_metrics:
                    val = model.avg_train_metrics[metric]
                    std = model.avg_train_metrics.get(f"{metric}_std", 0.0)
                    col = f"Training {metric.replace('train_', '').title()}"
                    row[col] = f"{val:.4f} (±{std:.4f})"
                    all_columns.add(col)

            for metric in test_metrics:
                if metric in model.avg_test_metrics:
                    val = model.avg_test_metrics[metric]
                    std = model.avg_test_metrics.get(f"{metric}_std", 0.0)
                    col = f"Test {metric.replace('test_', '').title()}"
                    row[col] = f"{val:.4f} (±{std:.4f})"
                    all_columns.add(col)
                    discovered_test_cols.add(col)
        else:
            for metric in train_metrics:
                if metric in getattr(model, "train_metrics", {}):
                    col = f"Training {metric.title()}"
                    row[col] = f"{model.train_metrics[metric]:.4f}"
                    all_columns.add(col)

            for metric in test_metrics:
                if metric in getattr(model, "test_metrics", {}):
                    col = f"Test {metric.title()}"
                    row[col] = f"{model.test_metrics[metric]:.4f}"
                    all_columns.add(col)
                    discovered_test_cols.add(col)

        rows.append(row)

    # --- Build DataFrame with union of columns discovered ---
    # Ensure deterministic ordering: Model first, then Training*, then Test*
    def _sort_key(c):
        if c == "Model": return (0, "")
        if c.startswith("Training "): return (1, c)
        if c.startswith("Test "): return (2, c)
        return (3, c)

    ordered_cols = sorted(all_columns, key=_sort_key)
    df = pd.DataFrame(rows, columns=ordered_cols).fillna("—")

    # --- Pretty prints (optional) ---
    print(f"\nModel Performance Comparison - {str(problem_type).title()}")
    # If any model had averaged results, show the max n_runs we saw
    max_runs = max([getattr(m, "n_runs", 1) for m in models.values()] or [1])
    if max_runs > 1:
        print(f"Results averaged over up to {max_runs} runs (where available).")
    print("=" * 100)
    print("\nDetailed Metrics:")
    print("-" * 100)
    print(df.to_string(index=False))

    # --- Best model per Test metric (works across mixed columns) ---
    print(f"\nBest Model by Test Metric ({str(problem_type).title()}):")
    print("-" * 100)

    # Helper: extract numeric from "0.9123 (±0.0100)" or "0.9123"
    def _extract_value(s):
        try:
            if isinstance(s, str) and "(" in s:
                return float(s.split("(")[0].strip())
            return float(s)
        except Exception:
            return np.nan

    for col in sorted(discovered_test_cols):
        vals = df[col].apply(_extract_value)
        metric_name = col.replace("Test ", "").lower()

        if metric_name in minimize_metrics:
            idx = vals.idxmin()
        else:
            idx = vals.idxmax()

        if idx is not None and not np.isnan(vals.iloc[idx]):
            print(f"{col:<22}: {df.iloc[idx]['Model']:<20} ({df.iloc[idx][col]})")

    # --- Feature importance (if available) ---
    print("\nTop 5 Features by Model:")
    print("-" * 100)
    for name, model in models.items():
        if hasattr(model, "avg_feature_importance") and getattr(model, "avg_feature_importance") is not None:
            afi = model.avg_feature_importance
            if hasattr(afi, "head"):
                print(f"\n{name} Top Features (averaged):")
                try:
                    print(afi.head().to_string(
                        index=False,
                        formatters={
                            "importance": "{:.6f}".format,
                            "importance_std": "{:.6f}".format
                        }
                    ))
                except Exception:
                    # Fallback if columns differ
                    print(afi.head().to_string(index=False))
        elif hasattr(model, "feature_importance") and getattr(model, "feature_importance") is not None:
            fi = model.feature_importance
            print(f"\n{name} Top Features:")
            try:
                print(fi.head().to_string())
            except Exception:
                print(str(fi))

        if hasattr(model, "explanation"):
            expl = getattr(model, "explanation")
            if expl:
                print(expl)

    # --- ROC curves (binary classification only), if helper available ---
    if problem_type == "classification" and is_binary and plot_roc_curves is not None:
        try:
            # Only test curves (include_train=False)
            # If your helper expects an n_runs, fall back to sample model's n_runs
            plot_roc_curves(models, getattr(sample_model, "n_runs", 1), include_train=False)
        except Exception as e:
            print(f"\nError creating ROC curves: {e}")

    return df
