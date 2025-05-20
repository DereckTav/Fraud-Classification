import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("ensemble_predictions_train.csv")

# Count group sizes
group_sizes = df.groupby(['predicted_type_final', 'rating']).size().reset_index(name='count')
print("Group sizes:\n", group_sizes)

# Step 1: Sample up to 1500 per group
TARGET_PER_GROUP = 1500
grouped = df.groupby(['predicted_type_final', 'rating'])

balanced_df = grouped.apply(
    lambda x: x.sample(n=min(len(x), TARGET_PER_GROUP), random_state=42)
).reset_index(drop=True)

# Step 2: Balance type 0 and type 1 rating distributions
# Count how many per rating we have in type 1
type1_counts = (
    balanced_df[balanced_df['predicted_type_final'] == 1]
    .groupby('rating')
    .size()
)

# Now sample type 0 to match those counts
type0_balanced_parts = []
for rating, count in type1_counts.items():
    subset = balanced_df[
        (balanced_df['predicted_type_final'] == 0) &
        (balanced_df['rating'] == rating)
    ]
    sampled = subset.sample(n=min(len(subset), count), random_state=42)
    type0_balanced_parts.append(sampled)

type0_balanced = pd.concat(type0_balanced_parts)

# Keep type 1 as-is (since it's already limited)
type1_balanced = balanced_df[balanced_df['predicted_type_final'] == 1]

# Final balanced set: type 0 now mirrors type 1
final_df = pd.concat([type0_balanced, type1_balanced]).reset_index(drop=True)

# Cross-check the distribution
print("\nFinal balanced distribution:")
print(pd.crosstab(final_df['predicted_type_final'], final_df['rating']))

# Step 3: Stratified split
# Ensure we get exactly 5000 rows in original_df
final_df['stratify_col'] = final_df['rating'].astype(str) + "_" + final_df['predicted_type_final'].astype(str)

original_df, automation_df = train_test_split(
    final_df,
    train_size=5000,
    random_state=42,
    stratify=final_df['stratify_col']
)


original_df = original_df.drop(columns=['stratify_col'])
automation_df = automation_df.drop(columns=['stratify_col'])

# Save
original_df.to_csv("original_reviews_train.csv", index=False)
automation_df.to_csv("for_automation_train.csv", index=False)

print(f"\nOriginal set size: {len(original_df)}")
print(f"Automation set size: {len(automation_df)}")
print("\n✅ Balanced, proportional datasets saved.")
