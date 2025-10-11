#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nested Cross-Validation + Backward SFS (KNN) for all ROI CSVs
--------------------------------------------------------------
Input:
  - ROI CSVs from script 1 (rows=subjects, first column 'subject_id', other columns = target-ROI features)
  - A labels CSV (id column + binary label column)

Process:
  - For each ROI CSV:
      * 4-fold outer CV (configurable)
      * Inner CV: backward SFS (mlxtend) with KNN and small grid over n_neighbors
      * 1-SE rule to choose a parsimonious subset
      * Evaluate on outer test split
      * Save fold selections, union/intersection, metrics

Outputs (per ROI):
  - <ROI>_Best_iterations.csv
  - <ROI>_Features_Union.csv
  - <ROI>_Features_Inter.csv (only if non-empty)
  - <ROI>_metrics.json

Also:
  - ALL_ROI_summary.csv (one row per ROI with mean test metrics)

Usage:
  python nested_cv_feature_selection.py \
    --roi-dir /path/to/All_ROIs_CSV \
    --labels /path/to/Harmo_Subj_Nico_list_reorder_as_train.csv \
    --label-col Group_Label \
    --id-col RF_Code \
    --roi-id-col subject_id \
    --out-dir /path/to/Results \
    --n-splits-outer 4 --n-splits-inner 4 --seed 42
"""

import argparse
import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef
)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# ------------------------------- CLI -----------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Nested CV + backward SFS (KNN) over ROI CSVs.")
    ap.add_argument("--roi-dir", required=True, help="Folder with per-ROI CSVs (from script 1).")
    ap.add_argument("--labels", required=True, help="CSV with subject labels.")
    ap.add_argument("--id-col", default="RF_Code", help="Subject ID column in labels CSV (e.g., RF_Code).")
    ap.add_argument("--roi-id-col", default="subject_id", help="Subject ID column in ROI CSVs (default: subject_id).")
    ap.add_argument("--label-col", required=True, help="Binary label column in labels CSV (0/1).")
    ap.add_argument("--out-dir", required=True, help="Output directory for results.")
    ap.add_argument("--n-splits-outer", type=int, default=4)
    ap.add_argument("--n-splits-inner", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pattern", default="*.csv", help="Glob pattern for ROI files (default: *.csv).")
    return ap.parse_args()


# ----------------------------- Utilities -------------------------------------
def evaluate_test(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
    }


def fit_sfs_choose_subset(X_tr, y_tr, n_splits_inner, seed):
    """
    Run backward SFS with a KNN estimator and small GridSearch over n_neighbors.
    Returns:
      - res_df: full SFS metric path
      - chosen_row: the row picked by 1-SE rule
      - chosen_names: list of selected feature names
      - best_est: the best GridSearch estimator (Pipeline('sfs', 'knn'))
    """
    base_knn = KNeighborsClassifier()
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)

    sfs = SFS(
        base_knn,
        k_features=(1, X_tr.shape[1]),
        forward=False,
        floating=False,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1,
        verbose=0,
    )

    pipe = Pipeline([("sfs", sfs), ("knn", KNeighborsClassifier())])

    param_grid = {
        "sfs__estimator__n_neighbors": [2, 3, 4],
        "knn__n_neighbors": [2, 3, 4],
    }

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv, verbose=0, n_jobs=-1)
    grid.fit(X_tr, y_tr)

    sfs_step = grid.best_estimator_.named_steps["sfs"]
    res_df = pd.DataFrame.from_dict(sfs_step.get_metric_dict()).T
    res_df["avg_score"] = pd.to_numeric(res_df["avg_score"])
    res_df["std_err"] = pd.to_numeric(res_df["std_err"])
    res_df["feature_count"] = res_df["feature_idx"].apply(len)

    best_avg = res_df["avg_score"].max()
    se_at_best = res_df.loc[res_df["avg_score"] == best_avg, "std_err"].mean()
    threshold = best_avg - se_at_best

    parsimonious = res_df[res_df["avg_score"] >= threshold].copy()
    chosen_row = parsimonious.loc[parsimonious["feature_count"].idxmin()]
    chosen_names = list(chosen_row["feature_names"])

    return res_df, chosen_row, chosen_names, grid.best_estimator_


# ------------------------------- Main ----------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    lab = pd.read_csv(args.labels)
    if args.id_col not in lab.columns or args.label_col not in lab.columns:
        raise ValueError(f"Labels file must contain '{args.id_col}' and '{args.label_col}'")
    lab = lab[[args.id_col, args.label_col]].copy()
    lab[args.id_col] = lab[args.id_col].astype(str)
    id2y = dict(zip(lab[args.id_col], lab[args.label_col]))

    roi_files = sorted(glob.glob(os.path.join(args.roi_dir, args.pattern)))
    if not roi_files:
        print(f"[WARN] No ROI CSVs in {args.roi_dir} matching {args.pattern}")
        return

    summary_rows = []

    for fpath in roi_files:
        roi_name = Path(fpath).stem
        print(f"\n=== ROI: {roi_name} ===")
        df = pd.read_csv(fpath)

        if args.roi_id_col not in df.columns:
            print(f"[SKIP] Missing id column '{args.roi_id_col}' in {fpath}")
            continue

        # Align subjects with labels
        ids = df[args.roi_id_col].astype(str).values
        X = df.drop(columns=[args.roi_id_col]).copy()

        y = np.array([id2y.get(sid, np.nan) for sid in ids])
        keep = ~pd.isna(y)
        dropped = int((~keep).sum())
        if dropped > 0:
            print(f"[INFO] Dropping {dropped} subjects without labels for {roi_name}.")

        X = X.loc[keep].reset_index(drop=True)
        y = y[keep].astype(int)
        ids_kept = np.array(ids)[keep]

        if len(np.unique(y)) < 2 or X.shape[0] < max(args.n_splits_outer, args.n_splits_inner):
            print(f"[SKIP] Not enough data or class variety for {roi_name}.")
            continue

        skf_outer = StratifiedKFold(n_splits=args.n_splits_outer, shuffle=True, random_state=args.seed)
        fold_summaries, test_metrics = [], []

        for fold, (tr, te) in enumerate(skf_outer.split(X, y), start=1):
            print(f"[Fold {fold}/{args.n_splits_outer}] train={len(tr)} test={len(te)}")

            X_tr, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
            y_tr, y_te = y[tr], y[te]

            # Inner: SFS + GridSearch
            res_df, chosen_row, chosen_names, best_est = fit_sfs_choose_subset(
                X_tr, y_tr, n_splits_inner=args.n_splits_inner, seed=args.seed
            )

            # Final KNN on chosen subset
            final_knn = best_est.named_steps["knn"]
            X_tr_sel, X_te_sel = X_tr[chosen_names], X_te[chosen_names]
            final_knn.fit(X_tr_sel, y_tr)
            y_pred = final_knn.predict(X_te_sel)

            m = evaluate_test(y_te, y_pred)
            m["fold"] = fold
            test_metrics.append(m)

            fold_summaries.append({
                "fold": fold,
                "feature_names": tuple(chosen_names),
                "avg_score": float(chosen_row["avg_score"]),
                "std_err": float(chosen_row["std_err"]),
                "feature_count": int(chosen_row["feature_count"]),
                "Test Acc": float(m["balanced_accuracy"]),
            })

        # Save fold selections
        fold_df = pd.DataFrame(fold_summaries)
        fold_path = out_dir / f"{roi_name}_Best_iterations.csv"
        fold_df.to_csv(fold_path, index=False)
        print(f"Saved: {fold_path}")

        # Feature union / intersection across folds
        sets = [set(s["feature_names"]) for s in fold_summaries] if fold_summaries else []
        if sets:
            union = sorted(set().union(*sets))
            inter = sorted(set.intersection(*sets)) if len(sets) > 1 else sorted(list(sets[0]))
        else:
            union, inter = [], []

        pd.DataFrame(union, columns=["Features Union"]).to_csv(out_dir / f"{roi_name}_Features_Union.csv", index=False)
        if inter:
            pd.DataFrame(inter, columns=["Features Inter"]).to_csv(out_dir / f"{roi_name}_Features_Inter.csv", index=False)

        # Aggregate metrics → JSON + summary row
        if test_metrics:
            tm = pd.DataFrame(test_metrics)
            mean = tm.drop(columns=["fold"]).mean(numeric_only=True).to_dict()
            std = tm.drop(columns=["fold"]).std(numeric_only=True).to_dict()
            agg = {"mean": mean, "std": std, "n_folds": int(tm.shape[0])}
            jpath = out_dir / f"{roi_name}_metrics.json"
            with open(jpath, "w") as f:
                json.dump(agg, f, indent=2)
            print(f"Saved: {jpath}")

            # One-line summary (balanced_accuracy mean ± std)
            summary_rows.append({
                "ROI": roi_name,
                **{f"mean_{k}": v for k, v in mean.items()},
                **{f"std_{k}": v for k, v in std.items()},
                "n_folds": agg["n_folds"],
            })

    # Study-wide summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_dir / "ALL_ROI_summary.csv"
        summary_df.sort_values(by="mean_balanced_accuracy", ascending=False).to_csv(summary_csv, index=False)
        print(f"\nSaved global summary: {summary_csv}")

    print("\nAll ROI files processed.")


if __name__ == "__main__":
    main()
