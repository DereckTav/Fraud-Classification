import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Let's first create a function to load and prepare both datasets
def prepare_combined_dataset(automation_path, original_path):
    """
    Load and combine datasets while creating a balanced dataset for fraud detection
    """
    # Load both datasets
    automation_df = pd.read_csv(automation_path)
    original_df = pd.read_csv(original_path)

    automation_df.drop(columns=['review'], inplace=True)
    automation_df.rename(columns={'generated_review': 'review'}, inplace=True)
    
    # Add Fraud column: 1 for automation dataset, 0 for original dataset
    automation_df['Fraud'] = 1
    original_df['Fraud'] = 0
    
    # Keep only relevant columns
    columns_to_keep = ['review', 'rating', 'predicted_type_final', 'Fraud']
    automation_df = automation_df[columns_to_keep]
    original_df = original_df[columns_to_keep]
    
    # Combine datasets
    combined_df = pd.concat([automation_df, original_df], ignore_index=True)
    
    # Create stratified sampling based on Fraud, rating, and polarity
    # This ensures balanced distribution across these factors
    combined_df['strata'] = combined_df['Fraud'].astype(str) + '_' + \
                           combined_df['rating'].astype(str) + '_' + \
                           combined_df['predicted_type_final'].astype(str)
    
    # Check distribution to understand how to balance
    print("Original distribution by strata:")
    print(combined_df['strata'].value_counts())
    
    # Calculate target counts for balanced dataset
    # For each stratum, determine minimum count that ensures balance
    strata_counts = combined_df['strata'].value_counts()
    fraud_strata = [s for s in strata_counts.index if s.startswith('1')]
    non_fraud_strata = [s for s in strata_counts.index if s.startswith('0')]
    
    # Determine minimum count per stratum to ensure balance between fraud/non-fraud
    min_count_per_stratum = min(
        min([strata_counts[s] for s in fraud_strata if s in strata_counts]),
        min([strata_counts[s] for s in non_fraud_strata if s in strata_counts])
    )
    
    # Create balanced dataset with equal representation
    balanced_samples = []
    for stratum in strata_counts.index:
        # Sample from each stratum
        stratum_samples = combined_df[combined_df['strata'] == stratum]
        
        # If stratum has more than minimum count, sample down to that
        if len(stratum_samples) > min_count_per_stratum:
            stratum_samples = stratum_samples.sample(n=min_count_per_stratum, random_state=42)
        
        balanced_samples.append(stratum_samples)
    
    # Combine all balanced samples
    balanced_df = pd.concat(balanced_samples, ignore_index=True)
    
    # Drop strata column as it's no longer needed
    balanced_df = balanced_df.drop('strata', axis=1)
    
    # Check final distribution
    print("\nFinal distribution:")
    print(f"Fraud: {balanced_df['Fraud'].value_counts()}")
    print(f"Ratings: {balanced_df['rating'].value_counts()}")
    print(f"Polarity: {balanced_df['predicted_type_final'].value_counts()}")
    
    # Shuffle the final dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

# Path to your datasets
automation_path = "finished_train.csv"
original_path = "original_reviews_train.csv"

# Create the balanced dataset
balanced_df = prepare_combined_dataset(automation_path, original_path)

# Save the datasets
balanced_df.to_csv("train.csv", index=False)

print(f"Saved balanced datasets with {len(balanced_df)} training samples")