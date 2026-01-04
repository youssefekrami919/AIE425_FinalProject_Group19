# ============================================================================ 
# Group: 19
# Team Members:
# - Youssef Ekrami Elsayed 
# - Abdelrahman Mohamed Negm
# - Hagar Mohamed Badawy
# - Dareen Ashraf Mosa
# ============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Define relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "SECTION1_DimensionalityReduction", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "SECTION1_DimensionalityReduction", "results")
CODE_DIR = os.path.join(BASE_DIR, "SECTION1_DimensionalityReduction", "code")

DATASET_PATH = os.path.join(DATA_DIR, "ratings.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots", "SVD plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables", "SVD")

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Helper function for saving CSV results
def save_csv(data, filename):
    data.to_csv(os.path.join(TABLES_DIR, filename), index=False)
    print(f"Saved {filename} to {TABLES_DIR}")

# Helper function for saving plots
def save_plot(figure, filename):
    figure.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Saved plot {filename} to {PLOTS_DIR}")

# 1.1. Load and prepare data
ratings_df = pd.read_csv(DATASET_PATH)
ratings_df = ratings_df[["userId", "movieId", "rating"]]
ratings_matrix = ratings_df.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

R = ratings_matrix.values  
R_subset_small = R[:500, :500]  



# 1.2. Calculate the average rating for each item (ri)
item_mean_ratings = ratings_matrix.replace(0, pd.NA).mean(axis=0)
item_mean_ratings_df = item_mean_ratings.reset_index()
item_mean_ratings_df.columns = ["movieId", "average_rating"]

print("First 10 item average ratings:")
print(item_mean_ratings_df.head(10))

save_csv(item_mean_ratings_df, "1_2_item_average_ratings.csv")

# 1.3. Apply mean-filling: replace missing ratings with the item's average rating
ratings_mean_filled = ratings_matrix.copy()

ratings_mean_filled = ratings_mean_filled.apply(
    lambda col: col.replace(0, item_mean_ratings[col.name]),
    axis=0
)

print("Ratings matrix after mean-filling (first 5 users, first 5 movies):")
print(ratings_mean_filled.iloc[:5, :5])


# 1.4. Verify matrix completeness (no missing values)
missing_count = ratings_mean_filled.isna().sum().sum()
print(f"Number of missing values in the matrix: {missing_count}")

# 2. Full SVD Decomposition
ratings_subset = ratings_mean_filled.iloc[:200, :300]
R_subset = ratings_subset.values

# Perform SVD on the subset (Full SVD)
U_full, sigma_full, Vt_full = np.linalg.svd(R_subset, full_matrices=False)

# Now you can use U_full
k = 100  # Choose optimal k based on your elbow curve analysis
U_k = U_full[:, :k]  # Select the first k components
Sigma_k = np.diag(sigma_full[:k])  # Create diagonal matrix for singular values
Vt_k = Vt_full[:k, :]  # Select the first k components of Vt

# Reconstruct the rating matrix using truncated SVD
R_svd_hat = U_k @ Sigma_k @ Vt_k

# 2. Compute prediction metrics (MAE, RMSE) for SVD
def prediction_metrics(R_true, R_hat):
    mask = R_true > 0  # Only consider the observed ratings
    mae = np.mean(np.abs(R_true[mask] - R_hat[mask]))  # Mean Absolute Error
    rmse = np.sqrt(np.mean((R_true[mask] - R_hat[mask]) ** 2))  # Root Mean Squared Error
    return mae, rmse

# Calculate prediction metrics for the SVD reconstruction
svd_mae, svd_rmse = prediction_metrics(R_subset, R_svd_hat)

# Print the results
print(f"SVD MAE: {svd_mae}")
print(f"SVD RMSE: {svd_rmse}")

# Saving the matrix results
np.save(os.path.join(TABLES_DIR, "2_2_U_matrix.npy"), U_full)
np.save(os.path.join(TABLES_DIR, "2_2_Sigma_matrix.npy"), np.diag(sigma_full))
np.save(os.path.join(TABLES_DIR, "2_2_Vt_matrix.npy"), Vt_full)

# 2.2. Calculate and save eigenvalues and eigenvectors
eigenvalues = sigma_full ** 2  # Eigenvalues based on sigma
eigenvectors = Vt_full.T  # Eigenvectors from Vt
eigenpairs = list(zip(eigenvalues, eigenvectors.T))

# Save eigenvalues
pd.DataFrame(eigenvalues, columns=["eigenvalue"]).to_csv(os.path.join(TABLES_DIR, "2_2_eigenvalues.csv"), index=False)

# 2.3. Verify orthogonality
UTU = U_full.T @ U_full
V = Vt_full.T
VTV = V.T @ V
I_U = np.eye(UTU.shape[0])
I_V = np.eye(VTV.shape[0])

deviation_U = np.linalg.norm(UTU - I_U)
deviation_V = np.linalg.norm(VTV - I_V)
print(f"Deviation from orthogonality (U): {deviation_U}")
print(f"Deviation from orthogonality (V): {deviation_V}")

# 2.4. Visualize singular values
plt.figure()
plt.plot(sigma_full)
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.title("Singular Values (Descending Order)")
save_plot(plt, "2_4_singular_values_plot.png")


# 3. Truncated SVD (Low-Rank Approximation)
k_values = [5, 20, 50, 100]
mae_values, rmse_values = [], []

for k in k_values:
    U_k = U_full[:, :k]
    Sigma_k = np.diag(sigma_full[:k])
    Vt_k = Vt_full[:k, :]

    R_k = U_k @ Sigma_k @ Vt_k
    mae = np.mean(np.abs(R_subset - R_k))
    rmse = np.sqrt(np.mean((R_subset - R_k) ** 2))

    mae_values.append(mae)
    rmse_values.append(rmse)

    print(f"k = {k}, MAE = {mae}, RMSE = {rmse}")

# Save the reconstruction error values
error_df = pd.DataFrame({
    "k": k_values,
    "MAE": mae_values,
    "RMSE": rmse_values
})
save_csv(error_df, "3_2_reconstruction_errors.csv")

# 3.3. Plot reconstruction error vs. k (elbow curve for SVD)
plt.figure()
plt.plot(k_values, mae_values, marker='o', label='MAE')
plt.plot(k_values, rmse_values, marker='s', label='RMSE')
plt.xlabel("k (latent factors)")
plt.ylabel("Error")
plt.title("Reconstruction Error vs k")
plt.legend()
save_plot(plt, "3_3_reconstruction_error_vs_k.png")

# --- Comparative Analysis with PCA ---

# Ensure that the input matrix does not contain NaN values by filling them
imputer = SimpleImputer(strategy="mean")
R_pca_mean_filled_imputed = imputer.fit_transform(R_subset)  # Fill missing values in the dataset

# PCA (Mean filling approach)
pca = PCA(n_components=100)
R_pca_mean_hat = pca.inverse_transform(pca.fit_transform(R_pca_mean_filled_imputed))  # Reconstruct the matrix

# Calculate prediction metrics for PCA
pca_mean_mae, pca_mean_rmse = prediction_metrics(R_subset, R_pca_mean_hat)

# Print results
print(f"PCA Mean MAE: {pca_mean_mae}")
print(f"PCA Mean RMSE: {pca_mean_rmse}")

# Save prediction accuracy results
prediction_df = pd.DataFrame({
    "Method": ["SVD", "PCA (Mean Filling)"],
    "MAE": [svd_mae, pca_mean_mae],
    "RMSE": [svd_rmse, pca_mean_rmse]
})
save_csv(prediction_df, "5_2_prediction_accuracy.csv")

# 6. Latent Factor Interpretation
k_factors = 3
U_k = U_full[:, :k_factors]
S_k = sigma_full[:k_factors]
V_k = Vt_full[:k_factors, :]

for i in range(k_factors):
    print(f"\nLatent Factor {i+1}")
    print("-" * 30)

    top_users = np.argsort(np.abs(U_k[:, i]))[::-1][:5]
    top_items = np.argsort(np.abs(V_k[i, :]))[::-1][:5]

    print("Top Users (indices):", top_users)
    print("Top Items (indices):", top_items)

# Visualizing latent space (users and items)
user_coords = U_k[:, :2]
item_coords = V_k[:2, :].T

user_activity = np.count_nonzero(R_subset, axis=1)
item_popularity = np.count_nonzero(R_subset, axis=0)

plt.figure(figsize=(10, 7))
plt.scatter(user_coords[:, 0], user_coords[:, 1], c=user_activity, cmap="Blues", alpha=0.6, label="Users")
plt.scatter(item_coords[:, 0], item_coords[:, 1], c=item_popularity, cmap="Reds", alpha=0.6, marker="x", label="Items")
plt.colorbar(label="Activity / Popularity")
plt.xlabel("Latent Factor 1")
plt.ylabel("Latent Factor 2")
plt.title("Latent Space Projection (Users and Items)")
plt.legend()
save_plot(plt, "6_3_latent_space_projection.png")


# 7.1 Sensitivity to Missing Data
# Create user–item rating matrix
ratings_pivot = ratings_df.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    fill_value=0
)

missing_levels = [0.1, 0.3, 0.5, 0.7]

recon_errors = []
mae_scores = []

for miss in missing_levels:
    # Create random mask
    random_mask = np.random.rand(*R_subset.shape) > miss
    R_missing = R_subset * random_mask

    # Mean fill (keep existing logic)
    mean_val = np.nanmean(R_missing[R_missing > 0])
    R_filled = np.where(R_missing == 0, mean_val, R_missing)

    # SVD
    U, S, Vt = np.linalg.svd(R_filled, full_matrices=False)
    k = 20
    R_hat = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    # Evaluate only originally observed entries
    diff = (R_subset - R_hat)[random_mask]
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))

    recon_errors.append(rmse)
    mae_scores.append(mae)

plt.plot(missing_levels, recon_errors, marker='o', label='RMSE')
plt.plot(missing_levels, mae_scores, marker='s', label='MAE')
plt.xlabel("Missing Data Percentage")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
save_plot(plt, "7_1_missing_data_sensitivity.png")

### 7.2 Impact of Initialization
# Item mean
item_mean = np.true_divide(R.sum(axis=0), (R != 0).sum(axis=0))
R_item_mean = np.where(R == 0, item_mean, R)

# User mean
user_mean = np.true_divide(R.sum(axis=1), (R != 0).sum(axis=1))
R_user_mean = np.where(R == 0, user_mean[:, None], R)

def svd_predict(R_input, k=20):
    U, S, Vt = np.linalg.svd(R_input, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

R_item_hat = svd_predict(R_item_mean)
R_user_hat = svd_predict(R_user_mean)

mae_item = np.mean(np.abs(R[R > 0] - R_item_hat[R > 0]))
mae_user = np.mean(np.abs(R[R > 0] - R_user_hat[R > 0]))

mae_item, mae_user

### 8.1 Cold-Start User Simulation
user_counts = (R > 0).sum(axis=1)
eligible_users = np.where(user_counts > 20)[0]

cold_users = np.random.choice(eligible_users, 50, replace=False)

R_cold = R.copy()
cold_masks = {}

for u in cold_users:
    rated_items = np.where(R[u] > 0)[0]
    hide = np.random.choice(rated_items, int(0.8 * len(rated_items)), replace=False)
    cold_masks[u] = hide
    R_cold[u, hide] = 0

### 8.2 Cold-Start Prediction
R_filled = np.where(R_cold == 0, np.nanmean(R_cold[R_cold > 0]), R_cold)
R_hat = svd_predict(R_filled)

errors = []

for u in cold_users:
    true = R[u, cold_masks[u]]
    pred = R_hat[u, cold_masks[u]]
    errors.extend(true - pred)

### 8.3 Cold vs Warm Start Performance
cold_mae = np.mean(np.abs(errors))
cold_rmse = np.sqrt(np.mean(np.array(errors) ** 2))

warm_mask = (R > 0) & (~np.isin(np.arange(R.shape[0])[:, None], cold_users))
warm_mae = np.mean(np.abs(R[warm_mask] - R_hat[warm_mask]))

cold_mae, cold_rmse, warm_mae

### 8.4 Cold-Start Mitigation Strategy
item_popularity = np.mean(R[R > 0])

alpha = 0.7
R_hybrid = alpha * R_hat + (1 - alpha) * item_popularity

hybrid_errors = []

for u in cold_users:
    hybrid_errors.extend(R[u, cold_masks[u]] - R_hybrid[u, cold_masks[u]])

np.mean(np.abs(hybrid_errors))