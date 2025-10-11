#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-ROI Connectivity Decomposition from .mat Files
-----------------------------------------------------
Loads all .mat connectivity matrices in a folder, each with ROI names stored
in the first row and first column (N x N matrix + labels), and produces one
CSV file per ROI containing the connectivity profile of all subjects.

Each CSV will have:
    - Rows: subjects
    - Columns: target ROIs (excluding self-connection)
    - File name: <ROI>.csv

Example:
    python decompose_connectivity_per_roi_from_mat.py \
        --mat-dir /path/to/mats \
        --out-dir /path/to/output \
        --var-name connectivity_matrix

If --var-name is omitted, the first 2D array in the .mat file is used.

-----------------------------------------------------
Author: Nicolo Pecco (adapted for GitHub reproducibility)
-----------------------------------------------------
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Convert connectivity .mat files (with ROI labels) to per-ROI CSVs.")
    ap.add_argument("--mat-dir", required=True, help="Folder containing .mat connectivity files.")
    ap.add_argument("--out-dir", required=True, help="Output folder for per-ROI CSVs.")
    ap.add_argument("--var-name", default=None, help="Optional: variable name of connectivity matrix inside .mat.")
    ap.add_argument("--keep-self", action="store_true", help="Keep self-connection column (default: dropped).")
    return ap.parse_args()


def load_matrix_with_labels(mat_path, var_name=None):
    """Load connectivity matrix and ROI names from first row and column."""
    data = loadmat(mat_path)

    # Select matrix variable
    if var_name is not None:
        M = data[var_name]
    else:
        # Auto-detect the first 2D array
        M = None
        for k, v in data.items():
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2:
                M = v
                break
        if M is None:
            raise ValueError(f"No valid 2D matrix found in {mat_path}")

    # Check for labels: first row & column
    if np.issubdtype(M.dtype, np.number):
        raise ValueError(f"Matrix in {mat_path} has no embedded ROI names; expected first row/column as strings.")

    # Convert MATLAB cell array of mixed numeric + strings
    M = np.array(M, dtype=object)
    roi_names = [str(x) for x in M[0, 1:]]  # first row, excluding top-left
    conn_values = M[1:, 1:].astype(float)

    return roi_names, conn_values


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(args.mat_dir, "*.mat")))
    if not mat_files:
        print(f"[WARN] No .mat files found in {args.mat_dir}")
        return

    subjects = []
    all_matrices = []
    roi_names_ref = None

    # ---- Load all subjects ----
    for f in mat_files:
        subj_id = Path(f).stem
        roi_names, conn = load_matrix_with_labels(f, var_name=args.var_name)

        if roi_names_ref is None:
            roi_names_ref = roi_names
        elif roi_names != roi_names_ref:
            raise ValueError(f"ROI names mismatch in {f}")

        all_matrices.append(conn)
        subjects.append(subj_id)
        print(f"Loaded {f} (shape: {conn.shape})")

    all_matrices = np.stack(all_matrices, axis=0)  # shape: (S, N, N)
    S, N, _ = all_matrices.shape
    print(f"\nLoaded {S} subjects, {N} ROIs each.")

    # ---- Generate one CSV per ROI ----
    for seed_idx, seed_name in enumerate(roi_names_ref):
        block = all_matrices[:, seed_idx, :]  # shape: (S, N)

        # drop self-connection unless --keep-self
        if not args.keep_self:
            mask = np.ones(N, dtype=bool)
            mask[seed_idx] = False
            block = block[:, mask]
            target_names = [n for i, n in enumerate(roi_names_ref) if i != seed_idx]
        else:
            target_names = roi_names_ref

        out_df = pd.DataFrame(block, columns=target_names)
        out_df.insert(0, "subject_id", subjects)

        safe_name = seed_name.replace("/", "_").replace(" ", "_")
        out_path = Path(args.out_dir) / f"{safe_name}.csv"
        out_df.to_csv(out_path, index=False)

        print(f"[{seed_idx+1:03d}/{N:03d}] Saved {out_path}")

    print("\n✅ Done. All ROI CSVs written to:", args.out_dir)


if __name__ == "__main__":
    main()
