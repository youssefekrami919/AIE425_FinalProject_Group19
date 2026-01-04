# SECTION 1 — Dimensionality Reduction (PCA / SVD)


Hager mohamed 
Section 1 part 3 (SVD & truncated SVD ) 

Dareen Ashraf
Section 1 part 3 ( Analysis) 

Abdelrahman negm
Section 1 part 2 ( PCA -MLE )

Youssef Ekrami 
Section 1 part 1 ( PCA - Mean filing )



This section contains experiments and scripts for PCA-based mean-filling, PCA with MLE, and SVD analysis on a MovieLens sample derived from the MovieLens 25M dataset.

Dataset

- Source: MovieLens 25M — https://grouplens.org/datasets/movielens/25m/
- Full dataset statistics:
	- Number of unique users: 162541
	- Number of unique items: 59047
	- Number of ratings: 25000095
	- Sparsity: 99.739516%

Subset used in this section

For the experiments in this section we select a reduced subset while preserving the project's selection criteria. The subset statistics used here are:
	- Number of unique users: 10025
	- Number of unique items: 24331
	- Number of ratings: 1500000
	- Sparsity: 99.385040%

Files and structure

- `code/` — scripts: `pca_mean_filling.py`, `pca_mle.py`, `svd_analysis.py`, `utils.py`.
- `data/` — dataset CSV files .
- `results/` — generated plots, tables .

How to prepare the dataset

1. Download the MovieLens 25M dataset from the link above and extract the files.
2. Place the relevant CSV(s) into `SECTION1_DimensionalityReduction/data/` or run the provided preprocessing (if any) to create the subset of 1,500,000 ratings following the project criteria.

Run instructions

1. Create and activate your Python virtual environment and install dependencies (see repository root `requirements.txt`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the PCA mean-filling experiment:

```powershell
python code\pca_mean_filling.py
```

3. Run the PCA (MLE) experiment:

```powershell
python code\pca_mle.py
```

4. Run the SVD analysis and reconstruction/prediction scripts:

```powershell
python code\svd_analysis.py
```




