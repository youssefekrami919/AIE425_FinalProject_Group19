# SECTION 2 — Domain Recommender (Collaborative / Content-based / Hybrid)

Hager mohamed
Section 2 part 3 (collaborative Filtering & hybrid approach )

Dareen Ashraf
Section 2 part 3 (  cold start Hybrid Approach)

Abdelrahman negm
Section 2 part 2 ( content - based Recommendation ) 

Youssef Ekrami 
Section 2  part 1 ( Domain Analysis and data Preparation)

This section implements recommender-system and tools for exploring results. The code is in `code/` and expects a dataset CSV placed in `data/`.

Dataset (synthetic)

- This section uses a generated synthetic dataset with the following summary:
	- Number of users: 5000
	- Number of items: 500
	- Number of ratings: 50000

- CSV columns :
	1. `interaction_id`
	2. `user_id`
	3. `item_id`
	4. `rating`
	5. `interaction_type`
	6. `view_time_sec`
	7. `age_group`
	8. `financial_goal`
	9. `financial_knowledge`
	10. `risk_tolerance`
	11. `occupation`
	12. `income_range`
	13. `title`
	14. `primary_topic`
	15. `subtopic`
	16. `difficulty`
	17. `content_type`
	18. `description`
	19. `summary`
	20. `content_length`
	21. `author`

Usage and roles

- The synthetic dataset is used by the content-based methods and by other parts of this section (collaborative and hybrid pipelines reference features from this dataset where appropriate).



Run instructions

1. Create and activate a Python virtual environment and install dependencies (see repository root `requirements.txt`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Preprocess / explore the dataset:

```powershell
python code\data_preprocessing.py
```

3. Run content-based recommendations (uses TF-IDF and encoders):

```powershell
python code\content_based.py
```

4. Run collaborative filtering and SVD experiments:

```powershell
python code\collaborative.py
```

5. Run the hybrid combination script:

```powershell
python code\hybrid.py
```




