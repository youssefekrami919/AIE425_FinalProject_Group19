# AIE425 Final Project (Group 19)



Hager mohamed 
Section 1 part 3 (SVD & truncated SVD ) 
Section 2 part 3 (collaborative Filtering & hybrid approach )

Dareen Ashraf
Section 1 part 3 ( Analysis) 
Section 2 part 3 (  cold start Hybrid Approach)

Abdelrahman negm
Section 1 part 2 ( PCA -MLE )
Section 2 part 2 ( content - based Recommendation ) 

Youssef Ekrami 
Section 1 part 1 ( PCA - Mean filing )
Section 2  part 1 ( Domain Analysis and data Preparation)




This repository contains two main parts of the project. Each part includes its own code, dataset, results and a dedicated README with instructions and dataset information.

- `SECTION1_DimensionalityReduction`: PCA and SVD experiments. See [SECTION1_DimensionalityReduction/README_SECTION1.md](SECTION1_DimensionalityReduction/README_SECTION1.md) for usage, dataset details, and how to reproduce the plots and tables.
- Source: MovieLens 25M — https://grouplens.org/datasets/movielens/25m/




- `SECTION2_DomainRecommender`: Recommender system (collaborative, content-based, hybrid). See [SECTION2_DomainRecommender/README_SECTION2.md](SECTION2_DomainRecommender/README_SECTION2.md) for usage, dataset details.

Quick setup

1. Create and activate a virtual environment (recommended).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Notes

- The repository root contains `requirements.txt` with the main Python dependencies used across both sections.
- Each section has a `data/` folder (datasets) and `results/` folder (plots, tables).

