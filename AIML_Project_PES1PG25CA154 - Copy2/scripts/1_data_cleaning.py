import pandas as pd
import numpy as np
import os

def clean_and_engineer_features():
    input_path = 'data/NAS_learning_outcomes_district.csv'
    output_path = 'artifacts/engineered_features.csv'
    
    print("⚡ Starting Step 2 & 3: Advanced Data Cleaning & Feature Engineering...")
    if not os.path.exists(input_path):
        print(f"❌ Error: Cannot find input file at {input_path}")
        return

    df = pd.read_csv(input_path)
    
    # Clean text anomalies
    df['subject'] = df['subject'].str.strip().str.lower()
    df['district_name'] = df['district_name'].str.strip()
    df['state_name'] = df['state_name'].str.strip()
    
    # Calculate a safe universal global default score from the raw averages
    global_score_default = float(df['avg'].median())
    if pd.isna(global_score_default):
        global_score_default = 50.0  # Safe absolute fallback
    
    # FEATURE ENGINEERING: Pivot multi-row raw data into district profiles
    print("Creating pivot table for subject-wise performance per grade...")
    print("\n===== SUBJECT VALUES =====")
    print(sorted(df['subject'].unique()))
    pivot_df = df.pivot_table(
        index=['district_name', 'state_name'], 
        columns=['grade', 'subject'], 
        values='avg', 
        aggfunc='mean'
    )
    
    # Flatten multi-index column headers
    pivot_df.columns = [f"grade_{col[0]}_{col[1]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    print("\n===== SCIENCE COLUMN CHECK =====")

    if 'grade_8_sci' in pivot_df.columns:
        print("grade_8_sci exists")
        print("Nulls:", pivot_df['grade_8_sci'].isna().sum())
        print("Non Nulls:", pivot_df['grade_8_sci'].notna().sum())
    else:
        print("grade_8_sci missing completely")
    # Map out exactly the 10 engineered features for early grades
    feature_mapping = {
        'grade_3_language': 'g3_lang', 'grade_3_math': 'g3_math', 'grade_3_evs': 'g3_evs',
        'grade_5_language': 'g5_lang', 'grade_5_math': 'g5_math', 'grade_5_evs': 'g5_evs',
        'grade_8_language': 'g8_lang', 'grade_8_math': 'g8_math', 'grade_8_sci': 'g8_sci', 'grade_8_sst': 'g8_sst'
    }
    
    # Defensive Safeguard: Ensure all 10 expected columns physically exist post-pivot
    for col in feature_mapping.keys():
        if col not in pivot_df.columns:
            pivot_df[col] = np.nan
    
    # Track target columns (Class 10 scores) to compute target metric
    target_cols = [col for col in pivot_df.columns if 'grade_10' in col]
    
    # Drop rows where target metric cannot be computed
    pivot_df = pivot_df.dropna(subset=target_cols, how='all')
    pivot_df['grade_10_avg_performance'] = pivot_df[target_cols].mean(axis=1)
    
    # Extract and filter target feature columns
    final_cols = ['district_name', 'state_name', 'grade_10_avg_performance'] + list(feature_mapping.keys())
    final_df = pivot_df[final_cols].rename(columns=feature_mapping)
    
    feature_cols = list(feature_mapping.values())
    print(f"Engineered Features Structure: {len(feature_cols)} core performance variables localized.")

    # ==========================================
    # IMPUTE ORIGINAL 10 FEATURES FIRST
    # ==========================================

    print("\n===== NULL COUNT BEFORE IMPUTATION =====")

    for col in feature_cols:
        print(col, final_df[col].isna().sum())

        # Tier 1: State median
        final_df[col] = final_df.groupby('state_name')[col].transform(
            lambda x: x.fillna(x.median())
        )

        # Tier 2: Row mean
        final_df[col] = final_df[col].fillna(
            final_df[feature_cols].mean(axis=1)
        )

        # Tier 3: Global fallback
        final_df[col] = final_df[col].fillna(
            global_score_default
        )

    # ==========================================
    # NOW CREATE ADVANCED FEATURES
    # ==========================================

    final_df['g3_avg'] = final_df[
        ['g3_lang', 'g3_math', 'g3_evs']
    ].mean(axis=1)

    final_df['g5_avg'] = final_df[
        ['g5_lang', 'g5_math', 'g5_evs']
    ].mean(axis=1)

    final_df['g8_avg'] = final_df[
        ['g8_lang', 'g8_math', 'g8_sci', 'g8_sst']
    ].mean(axis=1)

    final_df['growth_g3_g5'] = (
        final_df['g5_avg'] - final_df['g3_avg']
    )

    final_df['growth_g5_g8'] = (
        final_df['g8_avg'] - final_df['g5_avg']
    )

    final_df['performance_std'] = final_df[
        ['g3_avg', 'g5_avg', 'g8_avg']
    ].std(axis=1)

    # Safety check
    new_features = [
    'g3_avg',
    'g5_avg',
    'g8_avg',
    'growth_g3_g5',
    'growth_g5_g8',
    'performance_std'
    ]

    for col in new_features:
        final_df[col] = final_df[col].fillna(
            final_df[col].median()
    )

    print("✅ Advanced growth and performance features created.")
    # ADVANCED MULTI-TIER IMPUTATION PIPELINE (Eliminates NaNs and Zero Variance)
    print("\n===== NULL COUNT BEFORE IMPUTATION =====")

    for col in feature_cols:
        print(col, final_df[col].isna().sum())
        # Tier 1: Fill using state-wise median performance benchmark
        final_df[col] = final_df.groupby('state_name')[col].transform(lambda x: x.fillna(x.median()))
        # Tier 2: Row-wise cross-feature imputation (use other grade averages of same district)
        final_df[col] = final_df[col].fillna(final_df[feature_cols].mean(axis=1))
        # Tier 3: Universal fallback global benchmark
        final_df[col] = final_df[col].fillna(global_score_default)
        
    # Remove structural duplicates
    final_df = final_df.drop_duplicates()
    
    # Outlier handling via IQR capping boundaries
    print("Handling outliers via IQR capping boundaries...")
    for col in feature_cols:
        Q1 = final_df[col].quantile(0.25)
        Q3 = final_df[col].quantile(0.75)
        IQR = Q3 - Q1
        # Only cap if a valid spread exists
        if IQR > 0:
            final_df[col] = np.clip(final_df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
    os.makedirs('artifacts', exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print("✅ Data Cleaned and 10 Core Features Engineered successfully with zero NaNs!")
    print("\n===== NULL COUNT AFTER FEATURE ENGINEERING =====")

    for col in final_df.columns:
        nulls = final_df[col].isna().sum()
        if nulls > 0:
            print(col, nulls)
if __name__ == "__main__":
    clean_and_engineer_features()