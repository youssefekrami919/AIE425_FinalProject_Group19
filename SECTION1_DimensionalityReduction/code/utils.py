# ============================================================================ 
# Group: 19
# Team Members:
# - Youssef Ekrami Elsayed 
# - Abdelrahman Mohamed Negm
# - Hagar Mohamed Badawy
# - Dareen Ashraf Mosa
# ============================================================================


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
# HELPER FUNCTION: PREDICT RATING USING PEERS & COVARIANCE
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
# HELPER FUNCTION: Save CSV to TABLES_DIR
# ============================================================================
def save_csv(data, filename):
    data.to_csv(os.path.join(TABLES_DIR, filename), index=False)
    print(f"Saved {filename} to {TABLES_DIR}")

# ============================================================================ 
# HELPER FUNCTION: Save Plot to PLOTS_DIR
# ============================================================================
def save_plot(figure, filename):
    figure.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Saved plot {filename} to {PLOTS_DIR}")

# ============================================================================ 
# HELPER FUNCTION: Compute Prediction Metrics (MAE, RMSE)
# ============================================================================
def prediction_metrics(R_true, R_hat):
    mask = R_true > 0  # Only consider the observed ratings
    mae = np.mean(np.abs(R_true[mask] - R_hat[mask]))  # Mean Absolute Error
    rmse = np.sqrt(np.mean((R_true[mask] - R_hat[mask]) ** 2))  # Root Mean Squared Error
    return mae, rmse

# ============================================================================ 
# HELPER FUNCTION: SVD Prediction (Low-Rank Reconstruction)
# ============================================================================
def svd_predict(R_input, k=20):
    U, S, Vt = np.linalg.svd(R_input, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
