import pandas as pd

# --- CONFIGURATION ---
ORIGINAL_CSV = "test/original_reviews_test.csv"
AUTOMATION_CSV = "test/for_automation_test.csv"

def summarize_distribution(df, col_name):
    print(f"\nDistribution for '{col_name}':")
    counts = df[col_name].value_counts(dropna=False).sort_index()
    print(counts)
    print(f"Total: {counts.sum()}")

def cross_tabulate(df, col1, col2):
    print(f"\nCross-tabulation between '{col1}' and '{col2}':")
    ctab = pd.crosstab(df[col1], df[col2], dropna=False)
    print(ctab)

def analyze_dataset(name, df):
    print(f"\n{'='*40}\nAnalysis for dataset: {name}\n{'='*40}")
    print(f"Total reviews: {len(df)}")

    if 'predicted_type_final' in df.columns:
        summarize_distribution(df, 'predicted_type_final')
    else:
        print("Warning: 'predicted_type_final' column not found.")

    if 'rating' in df.columns:
        summarize_distribution(df, 'rating')
    else:
        print("Warning: 'rating' column not found.")

    # If both columns exist, cross-tabulate
    if 'predicted_type_final' in df.columns and 'rating' in df.columns:
        cross_tabulate(df, 'predicted_type_final', 'rating')

def main():
    print("\nLoading datasets...")
    try:
        original_df = pd.read_csv(ORIGINAL_CSV)
        automation_df = pd.read_csv(AUTOMATION_CSV)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    analyze_dataset("Original Reviews", original_df)
    analyze_dataset("Automation Reviews", automation_df)

    print("\n" + "="*40)
    print("Combined Dataset Analysis")
    combined_df = pd.concat([original_df, automation_df], ignore_index=True)
    analyze_dataset("Combined Dataset", combined_df)

if __name__ == "__main__":
    main()
