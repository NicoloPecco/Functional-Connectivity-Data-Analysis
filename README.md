# 🧠 Fetal fMRI Nested SFS Pipeline
*A modular framework for per-ROI connectivity decomposition and nested feature selection using backward SFS (KNN)*  

---

## 🧩 Overview  

This repository provides a **two-step Python pipeline** designed for **connectivity-based machine learning** on small neuroimaging datasets — such as **fetal fMRI**, **neonatal MRI**, or other clinical studies affected by **data scarcity** and **high-dimensional connectivity features**.

The goal is to make model training **reproducible, robust, and interpretable**, even when:
- The **number of subjects** is limited (tens of samples), and  
- The **number of features** (connections) is extremely high (thousands of edges).  

To tackle this, the pipeline:
1. **Decomposes** full subject-wise connectivity matrices into *per-ROI connectivity profiles* (one CSV per region).
2. **Runs nested cross-validation** with **backward Sequential Feature Selection (SFS)** using a **K-Nearest Neighbors (KNN)** classifier, to identify the most discriminative connections for **binary classification** problems (e.g., CHD vs Control, IDH-mut vs WT, etc.).

---

## 🧰 Repository Structure  

Functional-Connectivity-Data-Analysis/

│

├── decompose_connectivity_per_roi_from_mat.py   # Step 1: ROI-wise decomposition

├── nested_cv_feature_selection.py               # Step 2: Nested CV + SFS classification

├── README.md                                    # You're reading it :)

└── example_data/                                # (optional) Synthetic demo matrices

---

## 💡 Why this pipeline  

Brain connectivity analyses often involve **highly multivariate data** — e.g., 100+ ROIs yield >10,000 pairwise connections.  
At the same time, most fetal and neonatal studies have **limited sample sizes**.  
Training standard ML models directly on full connectivity matrices leads to:
- ⚠️ **Overfitting**
- ⚠️ **Unstable feature importance**
- ⚠️ **Low interpretability**

This pipeline helps mitigate those issues by:
- Breaking down large matrices into smaller, interpretable *per-ROI* problems  
- Using **nested cross-validation** to reduce bias  
- Employing **backward SFS** to identify only the most relevant features  
- Reporting **union** and **intersection** of features across folds for stability analysis  

---

## 🧠 Step 1 — ROI Decomposition  

A folder of `.mat` files — one per subject — each containing a square **connectivity matrix** (N×N) with ROI labels in the **first row and first column**.  

Example MATLAB structure:

|     | ROI_1 | ROI_2 | ROI_3 | ... |
|-----|-------|-------|-------|-----|
| ROI_1 | 0.0 | 0.23 | 0.54 | ... |
| ROI_2 | 0.23| 0.0  | 0.18 | ... |

Each `.mat` file represents one subject.

**Usage**  

**Argument	Description**
--mat-dir	Folder containing .mat files

--out-dir	Output folder for ROI CSVs

--var-name	Name of matrix variable inside each .mat

Each output CSV will contain:
subject_id | ROI_2 | ROI_3 | ROI_4 | ... (The output - by default - does not contain self-connection)

## ⚙️ **Step 2 — Nested Cross-Validation & Feature Selection**
🔹 **Input**
The folder of ROI CSVs generated in Step 1, with an added binary label column (e.g., 0/1, Control/CHD, IDH-mut/WT).
Example structure:
subject_id	label	ROI_2	ROI_3	ROI_4	...
**Argument	Description**
--roi-dir	Folder containing ROI CSVs
--out-dir	Output folder for results
--n-splits-outer	Outer CV folds (default = 4)
--n-splits-inner	Inner CV folds for SFS (default = 4)
--seed	Random seed for reproducibility

📊 Outputs
For each ROI, the script produces:
File	Description
ROI_X_Best_iterations.csv	Best feature subset per outer fold
ROI_X_Features_Union.csv	Union of selected features across folds
ROI_X_Features_Inter.csv	Intersection (stable features) across folds
ROI_X_metrics.json	Mean ± SD of test metrics
ALL_ROI_summary.csv	Summary ranking of all ROIs by performance

📈 How it works internally
Outer CV (4 folds) → Splits subjects into train/test partitions.
Inner CV (4 folds) → Performs backward SFS to iteratively remove weak features using KNN accuracy.
1-SE Rule → Chooses the smallest subset whose CV accuracy is within one standard error of the best model.
Final Model → Trains KNN on the selected features and evaluates on held-out test fold.
Evaluated metrics:
Accuracy
F1 (macro + weighted)
Balanced Accuracy
Matthews Correlation Coefficient

🧾 Interpreting the results
Union file: all features ever selected across folds → broad feature importance.
Intersection file: features consistently selected across folds → stable biomarkers.
Summary file: ranks ROIs by mean balanced accuracy → identify most predictive regions.

🧑‍💻 Citation
If you use this pipeline, please cite:
Pecco, N. et al. (2025).
A modular Python framework for nested feature selection in connectomic studies.
Università Vita-Salute San Raffaele, Milan.

