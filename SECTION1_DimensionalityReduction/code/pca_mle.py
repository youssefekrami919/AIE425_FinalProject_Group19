# ============================================================================ 
# Group: 19
# Team Members:
# - Youssef Ekrami Elsayed 
# - Abdelrahman Mohamed Negm
# - Hagar Mohamed Badawy
# - Dareen Ashraf Mosa
# ============================================================================

# ============================================================================
# PCA with Maximum Likelihood Estimation (MLE)
# FULL VERSION – 500x500 COVARIANCE MATRIX WITH TARGET ITEMS
# ============================================================================

import os
import numpy as np
import pandas as pd

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "SECTION1_DimensionalityReduction", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "SECTION1_DimensionalityReduction", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

PCA_MLE_DIR = os.path.join(TABLES_DIR, "PCA_MLE")
os.makedirs(PCA_MLE_DIR, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading ratings data...")
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
ratings = ratings.head(1_500_000)

# Target items
I1 = 3589
I2 = 4309

# Target users (IMPORTANT)
target_users = [51, 1, 500]

print(f"Target Items: I1 = {I1}, I2 = {I2}")
print(f"Target Users: {target_users}")

# ============================================================================
# HELPER FUNCTION: MLE COVARIANCE (COMMON USERS ONLY)
# ============================================================================
def mle_covariance(item_i, item_j, ratings_df):
    data = ratings_df[ratings_df['movieId'].isin([item_i, item_j])]
    pivot = data.pivot(index='userId', columns='movieId', values='rating')

    ri = pivot.get(item_i)
    rj = pivot.get(item_j)

    mask = ri.notna() & rj.notna()
    if mask.sum() == 0:
        return 0.0

    return np.mean((ri[mask] - ri[mask].mean()) * (rj[mask] - rj[mask].mean()))

# ============================================================================
# STEP 1: SELECT TOP 500 MOVIES
# ============================================================================
print("\nSelecting top 500 movies...")

movie_counts = ratings['movieId'].value_counts()
top500_movies = movie_counts.head(500).index.tolist()

if I1 not in top500_movies:
    top500_movies[0] = I1
if I2 not in top500_movies:
    top500_movies[1] = I2

movies = top500_movies
print(f"Total movies selected: {len(movies)}")

# ============================================================================
# STEP 2: COMPUTE COVARIANCE MATRIX
# ============================================================================
print("\nComputing 500x500 covariance matrix...")

cov_matrix = pd.DataFrame(index=movies, columns=movies, dtype=float)

for i in movies:
    for j in movies:
        cov_matrix.loc[i, j] = round(mle_covariance(i, j, ratings), 6)

cov_matrix.index.name = "movieId"
cov_matrix.to_csv(os.path.join(PCA_MLE_DIR, "covariance_matrix_mle.csv"), float_format="%.15f")

print("Covariance matrix saved.")

# ============================================================================
# STEP 3: EXTRACT I1 & I2 COVARIANCE VECTORS
# ============================================================================
cov_I1_series = cov_matrix.loc[I1].drop(I1).fillna(0)
cov_I2_series = cov_matrix.loc[I2].drop(I2).fillna(0)

cov_I1_series.to_csv(os.path.join(PCA_MLE_DIR, "covariance_I1.csv"))
cov_I2_series.to_csv(os.path.join(PCA_MLE_DIR, "covariance_I2.csv"))

# ============================================================================
# STEP 4: TOP 5 & TOP 10 PEERS
# ============================================================================
top5_I1 = cov_I1_series.sort_values(ascending=False).head(5)
top10_I1 = cov_I1_series.sort_values(ascending=False).head(10)

top5_I2 = cov_I2_series.sort_values(ascending=False).head(5)
top10_I2 = cov_I2_series.sort_values(ascending=False).head(10)

top5_I1.to_csv(os.path.join(PCA_MLE_DIR, "top5_peers_I1.csv"))
top10_I1.to_csv(os.path.join(PCA_MLE_DIR, "top10_peers_I1.csv"))
top5_I2.to_csv(os.path.join(PCA_MLE_DIR, "top5_peers_I2.csv"))
top10_I2.to_csv(os.path.join(PCA_MLE_DIR, "top10_peers_I2.csv"))

# ============================================================================
# STEP 5: RATING PREDICTION FUNCTION
# ============================================================================
def predict_rating(user_id, target_item, peers, cov_series, ratings_df):
    numerator = 0.0
    denominator = 0.0

    for peer in peers.index:
        r = ratings_df[
            (ratings_df['userId'] == user_id) &
            (ratings_df['movieId'] == peer)
        ]

        if not r.empty:
            weight = cov_series.get(peer, 0.0)
            numerator += weight * r['rating'].values[0]
            denominator += abs(weight)

    if denominator == 0:
        fallback = ratings_df[ratings_df['movieId'] == target_item]['rating'].mean()
        if np.isnan(fallback):
            fallback = ratings_df['rating'].mean()
        return round(fallback, 4)

    return round(numerator / denominator, 4)

# ============================================================================
# STEP 6: PREDICTIONS (TOP 5)
# ============================================================================
predictions_top5 = []

for u in target_users:
    predictions_top5.append({
        "UserId": u,
        "Predicted_I1": predict_rating(u, I1, top5_I1, cov_I1_series, ratings),
        "Predicted_I2": predict_rating(u, I2, top5_I2, cov_I2_series, ratings)
    })

top5_df = pd.DataFrame(predictions_top5)
top5_df.to_csv(os.path.join(PCA_MLE_DIR, "predictions_top5.csv"), index=False)

# ============================================================================
# STEP 7: PREDICTIONS (TOP 10)
# ============================================================================
predictions_top10 = []

for u in target_users:
    predictions_top10.append({
        "UserId": u,
        "Predicted_I1": predict_rating(u, I1, top10_I1, cov_I1_series, ratings),
        "Predicted_I2": predict_rating(u, I2, top10_I2, cov_I2_series, ratings)
    })

top10_df = pd.DataFrame(predictions_top10)
top10_df.to_csv(os.path.join(PCA_MLE_DIR, "predictions_top10.csv"), index=False)

# ============================================================================
# STEP 8: TOP 5 vs TOP 10 COMPARISON
# ============================================================================
comparison = []

for i, u in enumerate(target_users):
    comparison.append({
        "UserId": u,
        "I1_Top5": top5_df.loc[i, 'Predicted_I1'],
        "I1_Top10": top10_df.loc[i, 'Predicted_I1'],
        "I1_Difference": abs(top10_df.loc[i, 'Predicted_I1'] - top5_df.loc[i, 'Predicted_I1']),
        "I2_Top5": top5_df.loc[i, 'Predicted_I2'],
        "I2_Top10": top10_df.loc[i, 'Predicted_I2'],
        "I2_Difference": abs(top10_df.loc[i, 'Predicted_I2'] - top5_df.loc[i, 'Predicted_I2'])
    })

comparison_df = pd.DataFrame(comparison)
comparison_df.to_csv(os.path.join(PCA_MLE_DIR, "comparison_top5_vs_top10.csv"), index=False)

# ============================================================================
# STEP 9: COMPARE PART 1 WITH PART 2
# ============================================================================
print("\nComparing Part 1 with PCA MLE results...")

part1_top5 = pd.read_csv(os.path.join(TABLES_DIR, "PCA_MeanFilling", "step8_9_predictions_top5_peers.csv"))
part1_top10 = pd.read_csv(os.path.join(TABLES_DIR, "PCA_MeanFilling", "step10_11_predictions_top10_peers.csv"))

compare_top5 = pd.merge(part1_top5, top5_df, on="UserId", suffixes=('_Part1', '_PCA'))
compare_top5["I1_Diff"] = (compare_top5["Predicted_I1_Part1"] - compare_top5["Predicted_I1_PCA"]).abs()
compare_top5["I2_Diff"] = (compare_top5["Predicted_I2_Part1"] - compare_top5["Predicted_I2_PCA"]).abs()

compare_top10 = pd.merge(part1_top10, top10_df, on="UserId", suffixes=('_Part1', '_PCA'))
compare_top10["I1_Diff"] = (compare_top10["Predicted_I1_Part1"] - compare_top10["Predicted_I1_PCA"]).abs()
compare_top10["I2_Diff"] = (compare_top10["Predicted_I2_Part1"] - compare_top10["Predicted_I2_PCA"]).abs()

compare_top5.to_csv(os.path.join(PCA_MLE_DIR, "comparison_part1_vs_pca_top5.csv"), index=False)
compare_top10.to_csv(os.path.join(PCA_MLE_DIR, "comparison_part1_vs_pca_top10.csv"), index=False)

print("\n=== ALL STEPS COMPLETED SUCCESSFULLY ✅ ===")
