# ============================================================================ 
# Group: 19
# Team Members:
# - Youssef Ekrami Elsayed 
# - Abdelrahman Mohamed Negm
# - Hagar Mohamed Badawy
# - Dareen Ashraf Mosa
# ============================================================================

# ============================================================================
# IMPORT SECTION
# ============================================================================
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# PATHS CONFIGURATION
# ============================================================================


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "SECTION1_DimensionalityReduction", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "SECTION1_DimensionalityReduction", "results")
CODE_DIR = os.path.join(BASE_DIR, "SECTION1_DimensionalityReduction", "code")

DATASET_PATH = os.path.join(DATA_DIR, "ratings.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots", "PCA_mean_filling_plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables", "Data_preparation")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# ============================================================================
# DATA LOADING AND INITIAL ANALYSIS
# ============================================================================
print("Loading and analyzing dataset...")
df = pd.read_csv(DATASET_PATH)
df_sample = df.head(1500000)

unique_users = df_sample['userId'].nunique()
unique_items = df_sample['movieId'].nunique()
num_ratings = len(df_sample)
sparsity = 1 - (num_ratings / (unique_users * unique_items))

print(f"Number of unique users: {unique_users}")
print(f"Number of unique items: {unique_items}")
print(f"Number of ratings: {num_ratings}")
print(f"Sparsity: {sparsity:.6%}")


# ============================================================================
# USER ANALYSIS SECTION
# ============================================================================

# ----------------------------------------------------------------------------
# User Rating Counts Analysis
# ----------------------------------------------------------------------------
print("\nAnalyzing user rating counts...")
user_rating_counts = df_sample['userId'].value_counts()
user_rating_counts_df = user_rating_counts.reset_index()
user_rating_counts_df.columns = ['userId', 'rating_count']

user_rating_counts_df = user_rating_counts_df.sort_values(
    by=['rating_count', 'userId'], 
    ascending=[True, True]
).reset_index(drop=True)

user_rating_counts_df.to_csv(
    os.path.join(TABLES_DIR, "user_rating_counts_full.csv"), 
    index=False
)

# ----------------------------------------------------------------------------
# Select Representative Users (U1, U2, U3)
# ----------------------------------------------------------------------------
print("\nSelecting representative users...")
total_ratings = num_ratings
user_percentages = (user_rating_counts_df['rating_count'] / total_ratings) * 100

U1 = U2 = U3 = None
for idx, row in user_rating_counts_df.iterrows():
    percent = user_percentages.iloc[idx]
    
    if percent <= 2 and U1 is None:
        U1 = (row['userId'], row['rating_count'], round(percent, 5))
    elif 2 < percent <= 5 and U2 is None:
        U2 = (row['userId'], row['rating_count'], round(percent, 5))
    elif 5 < percent <= 10 and U3 is None:
        U3 = (row['userId'], row['rating_count'], round(percent, 5))
    
    if U1 and U2 and U3:
        break

print("\nTarget Users:")
for user_type, data in [("U1 (<=2%)", U1), ("U2 (2-5%)", U2), ("U3 (5-10%)", U3)]:
    if data:
        print(f"{user_type}: {data[0]} - {data[2]}% (Ratings: {data[1]})")
    else:
        print(f"{user_type}: No user found")

# Manual override for missing users
U2 = 1
U3 = 500
print(f"\nUpdated: U1={U1}, U2={U2}, U3={U3}")


# ============================================================================
# PLOTTING SECTION - USER ANALYSIS
# ============================================================================

# ----------------------------------------------------------------------------
# Plot 1: Distribution of Ratings per User (Log Scale)
# ----------------------------------------------------------------------------
print("\nGenerating user analysis plots...")
plt.figure(figsize=(10, 6))
plt.hist(user_rating_counts_df['rating_count'], bins=100, color='#4c72b0', 
         edgecolor='black', log=True)
plt.title('Distribution of Number of Ratings per User (Log Scale)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Number of Ratings per User', fontsize=12)
plt.ylabel('Number of Users (Log Scale)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "user_rating_distribution_log.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------------
# Plot 2: Top 50 Active Users
# ----------------------------------------------------------------------------
top_n = 50
top_users = user_rating_counts.sort_values(ascending=False).head(top_n)
plt.figure(figsize=(12, 7))
sns.barplot(x=top_users.values, y=top_users.index.astype(str), palette="viridis")
plt.title(f'Top {top_n} Most Active Users by Number of Ratings', 
          fontsize=14, fontweight='bold')
plt.xlabel('Number of Ratings', fontsize=12)
plt.ylabel('User ID', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "top_50_active_users.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------------
# Plot 3: Cumulative Ratings Coverage
# ----------------------------------------------------------------------------
cumulative = user_rating_counts.sort_values(ascending=False).cumsum() / num_ratings * 100
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative)+1), cumulative, color='#d62728')
plt.title('Cumulative Percentage of Ratings Covered by Users\n(Sorted by Activity)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Number of Users (from most active to least)', fontsize=12)
plt.ylabel('Cumulative % of Total Ratings', fontsize=12)
plt.grid(True, alpha=0.3)

users_80 = (cumulative >= 80).idxmax() + 1
plt.axvline(x=users_80, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=80, color='gray', linestyle='--', alpha=0.7)
plt.text(users_80 + 1000, 70, f'{users_80:,} users cover 80% of ratings', 
         fontsize=11, color='darkred')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cumulative_ratings_coverage.png"), 
            dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# ITEM ANALYSIS SECTION
# ============================================================================

# ----------------------------------------------------------------------------
# Calculate Average Ratings for All Items
# ----------------------------------------------------------------------------
print("\nCalculating target items (lowest average rated movies)...")
item_avg_list = []
for movie_id, group in df_sample.groupby('movieId')['rating']:
    item_avg_list.append((movie_id, group.mean()))

# Find lowest rated items
I1 = min(item_avg_list, key=lambda x: x[1])
item_avg_list_without_i1 = [item for item in item_avg_list if item != I1]
I2 = min(item_avg_list_without_i1, key=lambda x: x[1])

print("Target Items:")
print(f"I1, Lowest Rated: {I1}")
print(f"I2, Second Lowest Rated: {I2}")

# Sort and save all items
item_avg_list_sorted = sorted(item_avg_list, key=lambda x: x[1])
full_items_df = pd.DataFrame(item_avg_list_sorted, 
                           columns=['movieId', 'average_rating'])
full_items_df['movieId'] = full_items_df['movieId'].astype(int)
full_items_df['average_rating'] = full_items_df['average_rating'].round(3)

full_items_df.to_csv(
    os.path.join(TABLES_DIR, "all_movies_average_ratings_sorted.csv"), 
    index=False
)


# ============================================================================
# PLOTTING SECTION - ITEM ANALYSIS
# ============================================================================

# ----------------------------------------------------------------------------
# Plot 4: Distribution of Average Movie Ratings
# ----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(full_items_df['average_rating'], bins=50, color='#9467bd', 
         edgecolor='black')
plt.title('Distribution of Average Movie Ratings', 
          fontsize=14, fontweight='bold')
plt.xlabel('Average Rating', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "movie_average_rating_distribution.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------------
# Plot 5: Top 20 Lowest Rated Movies
# ----------------------------------------------------------------------------
lowest_20 = full_items_df.head(20)
plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(lowest_20)-1, -1, -1), 
                lowest_20['average_rating'], color='#ff7f0e')

# Highlight I1 and I2
i1_index = lowest_20[lowest_20['movieId'] == I1[0]].index[0] - lowest_20.index[0]
i2_index = lowest_20[lowest_20['movieId'] == I2[0]].index[0] - lowest_20.index[0]
bars[i1_index].set_color('#d62728')
bars[i2_index].set_color('#d62728')

plt.title('Top 20 Lowest Rated Movies\n(I1 and I2 highlighted in red)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Average Rating', fontsize=12)
plt.yticks(range(len(lowest_20)-1, -1, -1), lowest_20['movieId'].astype(str))
plt.ylabel('Movie ID', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "top_20_lowest_rated_movies.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------------
# Plot 6: Cumulative Percentage of Movies by Average Rating
# ----------------------------------------------------------------------------
cumulative_count = np.arange(1, len(full_items_df) + 1) / len(full_items_df) * 100
plt.figure(figsize=(10, 6))
plt.plot(full_items_df['average_rating'], cumulative_count, color='#2ca02c')
plt.title('Cumulative Percentage of Movies by Average Rating\n(Sorted from Lowest to Highest)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Average Rating', fontsize=12)
plt.ylabel('Cumulative Percentage of Movies (%)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.axvline(x=I1[1], color='red', linestyle='--', alpha=0.8)
plt.axvline(x=I2[1], color='red', linestyle='--', alpha=0.8)
plt.text(I1[1] + 0.05, 10, f'I1: {I1[1]:.3f}', color='red', fontweight='bold')
plt.text(I2[1] + 0.05, 20, f'I2: {I2[1]:.3f}', color='red', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cumulative_movie_ratings.png"), 
            dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# USER CATEGORIZATION SECTION
# ============================================================================

# ----------------------------------------------------------------------------
# Categorize Users and Save Tables
# ----------------------------------------------------------------------------
print("\nCategorizing users by activity level...")
user_percentages = (user_rating_counts_df['rating_count'] / num_ratings) * 100

# Define categories
cold_users_df = user_rating_counts_df[user_percentages <= 2].copy()
cold_users_df['percentage'] = (cold_users_df['rating_count'] / num_ratings * 100).round(5)

medium_users_df = user_rating_counts_df[(user_percentages > 2) & (user_percentages <= 5)].copy()
medium_users_df['percentage'] = (medium_users_df['rating_count'] / num_ratings * 100).round(5)

rich_users_df = user_rating_counts_df[user_percentages > 10].copy()
rich_users_df['percentage'] = (rich_users_df['rating_count'] / num_ratings * 100).round(5)

# Save user category tables
cold_users_df.to_csv(os.path.join(TABLES_DIR, "cold_users_le_2percent.csv"), index=False)
medium_users_df.to_csv(os.path.join(TABLES_DIR, "medium_users_2_to_5percent.csv"), index=False)
rich_users_df.to_csv(os.path.join(TABLES_DIR, "rich_users_gt_10percent.csv"), index=False)


# ============================================================================
# PLOTTING SECTION - USER CATEGORIES
# ============================================================================

# ----------------------------------------------------------------------------
# Plot 7: Distribution of Users by Activity Level (Pie Chart)
# ----------------------------------------------------------------------------
category_counts = [len(cold_users_df), len(medium_users_df), len(rich_users_df)]
category_labels = ['Cold Users (<=2%)', 'Medium Users (2-5%)', 'Rich Users (>10%)']
category_colors = ['#1f77b4', '#ff7f0e', '#d62728']

plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_labels, autopct='%1.1f%%', 
        colors=category_colors, startangle=90)
plt.title('Distribution of Users by Activity Level', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "user_categories_distribution_pie.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------------
# Plot 8: Rating Count Distribution by User Category
# ----------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.hist(cold_users_df['rating_count'], bins=50, alpha=0.7, 
         label='Cold Users (<=2%)', color='#1f77b4')
if not medium_users_df.empty:
    plt.hist(medium_users_df['rating_count'], bins=20, alpha=0.7, 
             label='Medium Users (2-5%)', color='#ff7f0e')
if not rich_users_df.empty:
    plt.hist(rich_users_df['rating_count'], bins=10, alpha=0.7, 
             label='Rich Users (>10%)', color='#d62728')
plt.title('Rating Count Distribution by User Category', 
          fontsize=14, fontweight='bold')
plt.xlabel('Number of Ratings', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "user_categories_rating_distribution.png"), 
            dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# ITEM POPULARITY CATEGORIZATION SECTION
# ============================================================================

# ----------------------------------------------------------------------------
# Categorize Items by Popularity and Save Tables
# ----------------------------------------------------------------------------
print("\nCategorizing items by popularity...")
item_rating_counts = df_sample['movieId'].value_counts()
item_rating_counts_df = item_rating_counts.reset_index()
item_rating_counts_df.columns = ['movieId', 'rating_count']

item_rating_counts_df = item_rating_counts_df.sort_values(
    by=['rating_count', 'movieId'], 
    ascending=[True, True]
).reset_index(drop=True)

item_rating_counts_df['percentage'] = (item_rating_counts_df['rating_count'] / num_ratings * 100).round(5)

# Define item categories
low_pop_items_df = item_rating_counts_df[item_rating_counts_df['percentage'] <= 2].copy()
medium_pop_items_df = item_rating_counts_df[(item_rating_counts_df['percentage'] > 2) & 
                                           (item_rating_counts_df['percentage'] <= 5)].copy()
high_pop_items_df = item_rating_counts_df[item_rating_counts_df['percentage'] > 10].copy()

# Save item category tables
low_pop_items_df.to_csv(os.path.join(TABLES_DIR, "low_popularity_items_le_2percent.csv"), index=False)
medium_pop_items_df.to_csv(os.path.join(TABLES_DIR, "medium_popularity_items_2_to_5percent.csv"), index=False)
high_pop_items_df.to_csv(os.path.join(TABLES_DIR, "high_popularity_items_gt_10percent.csv"), index=False)


# ============================================================================
# PLOTTING SECTION - ITEM POPULARITY
# ============================================================================

# ----------------------------------------------------------------------------
# Plot 9: Distribution of Items by Popularity Level (Pie Chart)
# ----------------------------------------------------------------------------
item_category_counts = [len(low_pop_items_df), len(medium_pop_items_df), len(high_pop_items_df)]
item_category_labels = ['Low Popularity (<=2%)', 'Medium Popularity (2-5%)', 'High Popularity (>10%)']

plt.figure(figsize=(8, 8))
plt.pie(item_category_counts, labels=item_category_labels, autopct='%1.1f%%', 
        colors=category_colors, startangle=90)
plt.title('Distribution of Items by Popularity Level', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "item_popularity_distribution_pie.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------------
# Plot 10: Rating Count Distribution by Item Popularity Category
# ----------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.hist(low_pop_items_df['rating_count'], bins=50, alpha=0.7, 
         label='Low Popularity (<=2%)', color='#1f77b4')
if not medium_pop_items_df.empty:
    plt.hist(medium_pop_items_df['rating_count'], bins=20, alpha=0.7, 
             label='Medium Popularity (2-5%)', color='#ff7f0e')
if not high_pop_items_df.empty:
    plt.hist(high_pop_items_df['rating_count'], bins=10, alpha=0.7, 
             label='High Popularity (>10%)', color='#d62728')
plt.title('Rating Count Distribution by Item Popularity Category', 
          fontsize=14, fontweight='bold')
plt.xlabel('Number of Ratings', fontsize=12)
plt.ylabel('Number of Items', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "item_popularity_rating_distribution.png"), 
            dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# SUMMARY TABLES SECTION
# ============================================================================

# ----------------------------------------------------------------------------
# Create and Save Summary Statistics Table
# ----------------------------------------------------------------------------
summary_data = {
    "Metric": [
        "Total Ratings in Sample",
        "Unique Users",
        "Unique Movies",
        "Potential Matrix Size",
        "Sparsity (%)",
        "Average Ratings per User",
        "Average Ratings per Movie"
    ],
    "Value": [
        f"{num_ratings:,}",
        f"{unique_users:,}",
        f"{unique_items:,}",
        f"{unique_users * unique_items:,}",
        f"{sparsity:.6%}",
        f"{num_ratings / unique_users:.2f}",
        f"{num_ratings / unique_items:.2f}"
    ]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(TABLES_DIR, "dataset_summary_statistics.csv"), index=False)

# ----------------------------------------------------------------------------
# Create and Save Representative Users Table
# ----------------------------------------------------------------------------
rep_users_data = {
    "User Type": ["U1 (Cold/Light User)", "U2 (Medium User)", "U3 (Heavy User)"],
    "User ID": [U1[0] if U1 else "Not found", U2 if U2 else "Not found", U3 if U3 else "Not found"],
    "Rating Count": [U1[1] if U1 else "-", "-", "-"],
    "Percentage of Total Ratings": [f"{U1[2]}%" if U1 else "-", "-", "-"]
}
rep_users_df = pd.DataFrame(rep_users_data)
rep_users_df.to_csv(os.path.join(TABLES_DIR, "representative_users_U1_U2_U3.csv"), index=False)


# ============================================================================
# PCA MEAN-FILLING ANALYSIS SECTION
# ============================================================================

# ----------------------------------------------------------------------------
# Setup PCA Directories
# ----------------------------------------------------------------------------
PCA_TABLES_DIR = os.path.join(TABLES_DIR, "PCA_MeanFilling")
os.makedirs(PCA_TABLES_DIR, exist_ok=True)

print("\n" + "="*60)
print("STARTING PCA MEAN-FILLING ANALYSIS")
print("="*60)

# ----------------------------------------------------------------------------
# Step 1: Get Average Ratings for I1 and I2
# ----------------------------------------------------------------------------
print("\nStep 1: Getting average ratings for target items...")
I1_avg = round(full_items_df[full_items_df['movieId'] == I1[0]]['average_rating'].values[0], 4)
I2_avg = round(full_items_df[full_items_df['movieId'] == I2[0]]['average_rating'].values[0], 4)

step1_df = pd.DataFrame({
    'Item': ['I1', 'I2'],
    'MovieId': [I1[0], I2[0]],
    'Average_Rating': [I1_avg, I2_avg]
})
step1_df.to_csv(os.path.join(PCA_TABLES_DIR, "step1_target_items_average_ratings.csv"), index=False)

# ----------------------------------------------------------------------------
# Step 2: Mean-Filling for I1 and I2
# ----------------------------------------------------------------------------
print("\nStep 2: Mean-filling for target items...")
all_users = df_sample['userId'].unique()
rating_matrix = pd.DataFrame(index=all_users, columns=[I1[0], I2[0]], dtype=float)

for _, row in df_sample[df_sample['movieId'].isin([I1[0], I2[0]])].iterrows():
    rating_matrix.loc[row['userId'], row['movieId']] = row['rating']

rating_matrix[I1[0]] = rating_matrix[I1[0]].fillna(I1_avg)
rating_matrix[I2[0]] = rating_matrix[I2[0]].fillna(I2_avg)

step2_df = rating_matrix.copy()
step2_df.index.name = 'userId'
step2_df.columns = ['I1_Rating', 'I2_Rating']
step2_df = step2_df.round(4)
step2_df.to_csv(os.path.join(PCA_TABLES_DIR, "step2_mean_filled_ratings.csv"))

# ----------------------------------------------------------------------------
# Step 3: Calculate Average Rating After Mean-Filling
# ----------------------------------------------------------------------------
print("\nStep 3: Calculating average ratings after mean-filling...")
I1_avg_after = round(rating_matrix[I1[0]].mean(), 4)
I2_avg_after = round(rating_matrix[I2[0]].mean(), 4)

step3_df = pd.DataFrame({
    'Item': ['I1', 'I2'],
    'MovieId': [I1[0], I2[0]],
    'Average_Rating_After_MeanFilling': [I1_avg_after, I2_avg_after]
})
step3_df.to_csv(os.path.join(PCA_TABLES_DIR, "step3_average_ratings_after_mean_filling.csv"), index=False)

# ----------------------------------------------------------------------------
# Step 4: Calculate Difference from Mean
# ----------------------------------------------------------------------------
print("\nStep 4: Calculating difference from mean...")
diff_matrix = pd.DataFrame(index=all_users, columns=[I1[0], I2[0]], dtype=float)
diff_matrix[I1[0]] = rating_matrix[I1[0]] - I1_avg_after
diff_matrix[I2[0]] = rating_matrix[I2[0]] - I2_avg_after
diff_matrix = diff_matrix.round(4)

step4_df = diff_matrix.copy()
step4_df.index.name = 'userId'
step4_df.columns = ['I1_Diff', 'I2_Diff']
step4_df.to_csv(os.path.join(PCA_TABLES_DIR, "step4_rating_differences_from_mean.csv"))

# ============================================================================
# STEPS 5 & 6: COMPUTE COVARIANCE MATRIX FOR 502 ITEMS (FIRST 500 + I1 + I2)
# ============================================================================
print("\nStep 5 & 6: Computing covariance matrix for 502 items (first 500 + I1 + I2)...")

# Get the first 500 unique items from the dataset (no sorting)
first_500_items = df_sample['movieId'].unique()[:500]
print(f"Selected first {len(first_500_items)} unique items from dataset")

# Ensure I1 and I2 are included in our item list
all_items = list(first_500_items)

# Add I1 and I2 if they're not already in the first 500
if I1[0] not in all_items:
    all_items.append(I1[0])
    print(f"Added I1 (Movie {I1[0]}) to item list")
else:
    print(f"I1 (Movie {I1[0]}) already in first 500 items")

if I2[0] not in all_items:
    all_items.append(I2[0])
    print(f"Added I2 (Movie {I2[0]}) to item list")
else:
    print(f"I2 (Movie {I2[0]}) already in first 500 items")

print(f"Total items for covariance matrix: {len(all_items)} items")

# Get all users
all_users = df_sample['userId'].unique()
print(f"Total users: {len(all_users)} users")

# Initialize a DataFrame to store mean-filled ratings for all items
print("\nCreating mean-filled rating matrix for all items...")
rating_matrix_all = pd.DataFrame(index=all_users, columns=all_items, dtype=float)

# Fill the rating matrix with actual ratings or item means
for item in all_items:
    # Get actual ratings for this item
    item_ratings = df_sample[df_sample['movieId'] == item][['userId', 'rating']]
    
    # For I1 and I2, we need to ensure we use the correct average ratings
    # that were computed earlier (not recalculating from potentially sparse data)
    if item == I1[0]:
        # Use the precomputed average for I1 from Step 1
        item_mean = I1_avg
    elif item == I2[0]:
        # Use the precomputed average for I2 from Step 1
        item_mean = I2_avg
    else:
        # For other items, calculate mean from actual ratings
        if not item_ratings.empty:
            item_mean = item_ratings['rating'].mean()
        else:
            # If no ratings exist (unlikely), use global mean
            item_mean = df_sample['rating'].mean()
    
    # Fill the rating matrix
    # For I1 and I2: use actual ratings where available, otherwise use mean
    # For other items: use actual ratings where available, otherwise use mean
    for user in all_users:
        user_rating = item_ratings[item_ratings['userId'] == user]
        if not user_rating.empty:
            rating_matrix_all.loc[user, item] = user_rating['rating'].values[0]
        else:
            rating_matrix_all.loc[user, item] = item_mean

print(f"Rating matrix shape: {rating_matrix_all.shape}")
print(f"Memory usage: {rating_matrix_all.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Calculate mean for each item after mean-filling
item_means_after = rating_matrix_all.mean()
print(f"Mean ratings calculated for {len(item_means_after)} items")

# Check if I1 and I2 have non-zero values
print("\nChecking I1 and I2 values in rating matrix:")
print(f"I1 ({I1[0]}) - Number of non-zero values: {(rating_matrix_all[I1[0]] != 0).sum()}")
print(f"I1 ({I1[0]}) - Mean value: {rating_matrix_all[I1[0]].mean():.4f}")
print(f"I2 ({I2[0]}) - Number of non-zero values: {(rating_matrix_all[I2[0]] != 0).sum()}")
print(f"I2 ({I2[0]}) - Mean value: {rating_matrix_all[I2[0]].mean():.4f}")

# Calculate difference matrix (rating - mean)
print("\nCalculating difference from mean for all items...")
diff_matrix_all = rating_matrix_all.sub(item_means_after, axis=1)
print(f"Difference matrix shape: {diff_matrix_all.shape}")

# Check if I1 and I2 have non-zero differences
print(f"\nI1 ({I1[0]}) - Difference mean: {diff_matrix_all[I1[0]].mean():.4f}")
print(f"I1 ({I1[0]}) - Difference std: {diff_matrix_all[I1[0]].std():.4f}")
print(f"I2 ({I2[0]}) - Difference mean: {diff_matrix_all[I2[0]].mean():.4f}")
print(f"I2 ({I2[0]}) - Difference std: {diff_matrix_all[I2[0]].std():.4f}")

# Save difference matrix for reference
diff_matrix_path = os.path.join(PCA_TABLES_DIR, "difference_matrix_502_items.csv")
diff_matrix_all.round(4).to_csv(diff_matrix_path)
print(f"Difference matrix saved to: {diff_matrix_path}")

# Compute covariance matrix (502x502)
print("\nComputing covariance matrix (502x502)...")
print("This might take a moment for 502x502 matrix...")

# Using matrix multiplication for efficiency
n_users = len(all_users)
covariance_matrix_502 = (diff_matrix_all.T @ diff_matrix_all) / n_users

# Round to 4 decimal places
covariance_matrix_502 = covariance_matrix_502.round(4)

print(f"Covariance matrix shape: {covariance_matrix_502.shape}")
print(f"Memory usage: {covariance_matrix_502.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Save the full 502x502 covariance matrix
full_covariance_path = os.path.join(PCA_TABLES_DIR, "step5_6_covariance_matrix_502x502.csv")
covariance_matrix_502.to_csv(full_covariance_path)
print(f"\nFull 502x502 covariance matrix saved to: {full_covariance_path}")

# Verify I1 and I2 in covariance matrix
print("\nVerifying I1 and I2 in covariance matrix...")
if I1[0] in covariance_matrix_502.index and I2[0] in covariance_matrix_502.index:
    print(f"I1 (Movie {I1[0]}) and I2 (Movie {I2[0]}) are in the covariance matrix")
    
    # Get their positions in the matrix
    i1_position = list(covariance_matrix_502.index).index(I1[0])
    i2_position = list(covariance_matrix_502.index).index(I2[0])
    
    print(f"I1 (Movie {I1[0]}) is at position {i1_position} in the matrix")
    print(f"I2 (Movie {I2[0]}) is at position {i2_position} in the matrix")
    
    # Check self-covariance (variance) for I1 and I2
    print(f"\nSelf-covariance (variance) for I1: {covariance_matrix_502.loc[I1[0], I1[0]]:.6f}")
    print(f"Self-covariance (variance) for I2: {covariance_matrix_502.loc[I2[0], I2[0]]:.6f}")
    print(f"Covariance between I1 and I2: {covariance_matrix_502.loc[I1[0], I2[0]]:.6f}")
    
    # Get top 5 covariances for I1 and I2 (excluding themselves)
    cov_with_I1 = covariance_matrix_502[I1[0]].copy()
    cov_with_I1 = cov_with_I1.drop(I1[0])  # Remove self-covariance
    cov_with_I1 = cov_with_I1.sort_values(ascending=False)
    
    cov_with_I2 = covariance_matrix_502[I2[0]].copy()
    cov_with_I2 = cov_with_I2.drop(I2[0])  # Remove self-covariance
    cov_with_I2 = cov_with_I2.sort_values(ascending=False)
    
    print(f"\nTop 5 highest covariances for I1 (Movie {I1[0]}):")
    for i, (item, cov) in enumerate(cov_with_I1.head(5).items(), 1):
        print(f"  {i}. Movie {item}: Covariance = {cov:.4f}")
    
    print(f"\nTop 5 highest covariances for I2 (Movie {I2[0]}):")
    for i, (item, cov) in enumerate(cov_with_I2.head(5).items(), 1):
        print(f"  {i}. Movie {item}: Covariance = {cov:.4f}")
else:
    print(f"ERROR: I1 ({I1[0]}) or I2 ({I2[0]}) not found in covariance matrix!")

# Also compute and save summary statistics of the covariance matrix
print("\nComputing covariance matrix statistics...")
cov_stats = {
    'Metric': [
        'Matrix Size',
        'Total Elements',
        'Mean Covariance',
        'Std Deviation',
        'Min Covariance',
        'Max Covariance',
        'Diagonal Mean (Variance)',
        'Off-Diagonal Mean'
    ],
    'Value': [
        f"{covariance_matrix_502.shape[0]}x{covariance_matrix_502.shape[1]}",
        f"{covariance_matrix_502.size:,}",
        f"{covariance_matrix_502.values.mean():.6f}",
        f"{covariance_matrix_502.values.std():.6f}",
        f"{covariance_matrix_502.values.min():.6f}",
        f"{covariance_matrix_502.values.max():.6f}",
        f"{np.diag(covariance_matrix_502).mean():.6f}",
        f"{covariance_matrix_502.values[~np.eye(covariance_matrix_502.shape[0], dtype=bool)].mean():.6f}"
    ]
}

cov_stats_df = pd.DataFrame(cov_stats)
cov_stats_path = os.path.join(PCA_TABLES_DIR, "covariance_matrix_statistics.csv")
cov_stats_df.to_csv(cov_stats_path, index=False)
print(f"\nCovariance matrix statistics saved to: {cov_stats_path}")
print("\nCovariance Matrix Statistics:")
print(cov_stats_df.to_string(index=False))

print("\n" + "="*70)
print("STEP 5 & 6 COMPLETED: 502x502 COVARIANCE MATRIX COMPUTED")
print(f"  - Items: {len(all_items)} (500 from dataset + I1 + I2)")
print(f"  - Users: {len(all_users)}")
print(f"  - Matrix Size: {covariance_matrix_502.shape[0]} x {covariance_matrix_502.shape[1]}")
print("="*70)

# ----------------------------------------------------------------------------
# Step 7: Top 5 and Top 10 Peers
# ----------------------------------------------------------------------------
print("\nStep 7: Determining top peers...")
all_movies = df_sample['movieId'].unique()
movie_cov_I1 = []
movie_cov_I2 = []

for movie in all_movies:
    if movie in [I1[0], I2[0]]:
        continue
    
    movie_ratings = df_sample[df_sample['movieId'] == movie][['userId', 'rating']]
    movie_avg = movie_ratings['rating'].mean()
    
    movie_vector = pd.Series(index=all_users, dtype=float)
    for _, row in movie_ratings.iterrows():
        movie_vector.loc[row['userId']] = row['rating']
    movie_vector = movie_vector.fillna(movie_avg)
    
    movie_diff = movie_vector - movie_avg
    cov_with_I1 = round(np.sum(movie_diff * diff_matrix[I1[0]]) / len(diff_matrix), 4)
    cov_with_I2 = round(np.sum(movie_diff * diff_matrix[I2[0]]) / len(diff_matrix), 4)
    
    movie_cov_I1.append((movie, cov_with_I1))
    movie_cov_I2.append((movie, cov_with_I2))

# Sort and save top peers
movie_cov_I1_sorted = sorted(movie_cov_I1, key=lambda x: x[1], reverse=True)
movie_cov_I2_sorted = sorted(movie_cov_I2, key=lambda x: x[1], reverse=True)

top5_I1 = movie_cov_I1_sorted[:5]
top10_I1 = movie_cov_I1_sorted[:10]
top5_I2 = movie_cov_I2_sorted[:5]
top10_I2 = movie_cov_I2_sorted[:10]

pd.DataFrame(top5_I1, columns=['MovieId', 'Covariance_with_I1']).to_csv(
    os.path.join(PCA_TABLES_DIR, "step7_top5_peers_I1.csv"), index=False)
pd.DataFrame(top10_I1, columns=['MovieId', 'Covariance_with_I1']).to_csv(
    os.path.join(PCA_TABLES_DIR, "step7_top10_peers_I1.csv"), index=False)
pd.DataFrame(top5_I2, columns=['MovieId', 'Covariance_with_I2']).to_csv(
    os.path.join(PCA_TABLES_DIR, "step7_top5_peers_I2.csv"), index=False)
pd.DataFrame(top10_I2, columns=['MovieId', 'Covariance_with_I2']).to_csv(
    os.path.join(PCA_TABLES_DIR, "step7_top10_peers_I2.csv"), index=False)


# ----------------------------------------------------------------------------
# Steps 8 & 9: Predictions Using Top 5 Peers
# ----------------------------------------------------------------------------
print("\nStep 8 & 9: Computing predictions using top 5 peers...")
target_users = [U1[0], U2, U3]
top5_I1_movies = [movie for movie, _ in top5_I1]
top5_I2_movies = [movie for movie, _ in top5_I2]

predictions_top5 = []
for user in target_users:
    # Predict I1
    user_I1_peer_ratings = []
    for movie in top5_I1_movies:
        user_rating = df_sample[(df_sample['userId'] == user) & (df_sample['movieId'] == movie)]
        if not user_rating.empty:
            user_I1_peer_ratings.append(user_rating['rating'].values[0])
        else:
            movie_avg = df_sample[df_sample['movieId'] == movie]['rating'].mean()
            user_I1_peer_ratings.append(movie_avg)
    predicted_I1 = round(np.mean(user_I1_peer_ratings), 4)
    
    # Predict I2
    user_I2_peer_ratings = []
    for movie in top5_I2_movies:
        user_rating = df_sample[(df_sample['userId'] == user) & (df_sample['movieId'] == movie)]
        if not user_rating.empty:
            user_I2_peer_ratings.append(user_rating['rating'].values[0])
        else:
            movie_avg = df_sample[df_sample['movieId'] == movie]['rating'].mean()
            user_I2_peer_ratings.append(movie_avg)
    predicted_I2 = round(np.mean(user_I2_peer_ratings), 4)
    
    predictions_top5.append({
        'UserId': user,
        'Predicted_I1': predicted_I1,
        'Predicted_I2': predicted_I2
    })

step8_9_df = pd.DataFrame(predictions_top5)
step8_9_df.to_csv(os.path.join(PCA_TABLES_DIR, "step8_9_predictions_top5_peers.csv"), index=False)


# ----------------------------------------------------------------------------
# Steps 10 & 11: Predictions Using Top 10 Peers
# ----------------------------------------------------------------------------
print("\nStep 10 & 11: Computing predictions using top 10 peers...")
top10_I1_movies = [movie for movie, _ in top10_I1]
top10_I2_movies = [movie for movie, _ in top10_I2]

predictions_top10 = []
for user in target_users:
    # Predict I1
    user_I1_peer_ratings = []
    for movie in top10_I1_movies:
        user_rating = df_sample[(df_sample['userId'] == user) & (df_sample['movieId'] == movie)]
        if not user_rating.empty:
            user_I1_peer_ratings.append(user_rating['rating'].values[0])
        else:
            movie_avg = df_sample[df_sample['movieId'] == movie]['rating'].mean()
            user_I1_peer_ratings.append(movie_avg)
    predicted_I1 = round(np.mean(user_I1_peer_ratings), 4)
    
    # Predict I2
    user_I2_peer_ratings = []
    for movie in top10_I2_movies:
        user_rating = df_sample[(df_sample['userId'] == user) & (df_sample['movieId'] == movie)]
        if not user_rating.empty:
            user_I2_peer_ratings.append(user_rating['rating'].values[0])
        else:
            movie_avg = df_sample[df_sample['movieId'] == movie]['rating'].mean()
            user_I2_peer_ratings.append(movie_avg)
    predicted_I2 = round(np.mean(user_I2_peer_ratings), 4)
    
    predictions_top10.append({
        'UserId': user,
        'Predicted_I1': predicted_I1,
        'Predicted_I2': predicted_I2
    })

step10_11_df = pd.DataFrame(predictions_top10)
step10_11_df.to_csv(os.path.join(PCA_TABLES_DIR, "step10_11_predictions_top10_peers.csv"), index=False)



# ----------------------------------------------------------------------------
# Step 12: Comparison of Top 5 vs Top 10 Predictions
# ----------------------------------------------------------------------------
print("\nStep 12: Comparing Top 5 vs Top 10 predictions...")
comparison_data = []
for i, user in enumerate(target_users):
    comparison_data.append({
        'UserId': user,
        'I1_Top5': predictions_top5[i]['Predicted_I1'],
        'I1_Top10': predictions_top10[i]['Predicted_I1'],
        'I1_Difference': round(abs(predictions_top10[i]['Predicted_I1'] - predictions_top5[i]['Predicted_I1']), 4),
        'I2_Top5': predictions_top5[i]['Predicted_I2'],
        'I2_Top10': predictions_top10[i]['Predicted_I2'],
        'I2_Difference': round(abs(predictions_top10[i]['Predicted_I2'] - predictions_top5[i]['Predicted_I2']), 4)
    })

step12_df = pd.DataFrame(comparison_data)
step12_df.to_csv(os.path.join(PCA_TABLES_DIR, "step12_comparison_top5_vs_top10.csv"), index=False)


print("\n=== All Files Generated Successfully ===")
