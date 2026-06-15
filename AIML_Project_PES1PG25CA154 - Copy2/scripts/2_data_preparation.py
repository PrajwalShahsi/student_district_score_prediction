import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import joblib
from sklearn.preprocessing import LabelEncoder

def prepare_select_and_extract():
    df = pd.read_csv('artifacts/engineered_features.csv')
    
    # Establish categorical target classes (Low, Medium, High Performance)
    df['Performance_Class'] = pd.qcut(df['grade_10_avg_performance'], q=3, labels=['Low', 'Medium', 'High'])
    
    # Encode variables
    le_state = LabelEncoder()
    df['state_encoded'] = le_state.fit_transform(df['state_name'])
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(df['Performance_Class'])
    
    # Define our 10 engineered feature column space
    feature_cols = [

    # Original Features
    'g3_lang', 'g3_math', 'g3_evs',
    'g5_lang', 'g5_math', 'g5_evs',
    'g8_lang', 'g8_math', 'g8_sci', 'g8_sst',

    # Engineered Features
    'g3_avg',
    'g5_avg',
    'g8_avg',
    'growth_g3_g5',
    'growth_g5_g8',
    'performance_std',

    # Geographic Feature
    'state_encoded'
    ]
    X = df[feature_cols]
    
    # CRITICAL FOR ZERO DATA LEAKAGE: Split data BEFORE fitting Scaler, Selection or Extraction tools
    print("Splitting data into 80% Train and 20% Test sets to protect pipeline integrity...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 1. FEATURE SCALING
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. FEATURE SELECTION (Anova F-value evaluation to check significance)
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_train_scaled, y_train)
    feature_scores = selector.scores_.tolist()
    
    # 3. FEATURE EXTRACTION (Applying PCA to generate simplified, optimal dense components)
    #print("Executing Feature Extraction using Principal Component Analysis (PCA)...")
    #pca = PCA(n_components=5, random_state=42) # Compress 10 dimensions into 5 principal orthogonal features
    #X_train_extracted = pca.fit_transform(X_train_scaled)
    #X_test_extracted = pca.transform(X_test_scaled)
    
    #print(f"Extracted variance capture ratio: {np.sum(pca.explained_variance_ratio_)*100:.2f}% variance explained by 5 components.")
    print("Skipping PCA - using original engineered features.")

    X_train_extracted = X_train_scaled
    X_test_extracted = X_test_scaled
    # Save transformers and encoded mappings
    joblib.dump(scaler, 'artifacts/scaler.pkl')
    #joblib.dump(pca, 'artifacts/pca_extractor.pkl')
    joblib.dump(le_state, 'artifacts/le_state.pkl')
    joblib.dump(le_target, 'artifacts/le_target.pkl')
    
    # Convert numpy arrays to dataframes for training ingestion
    #pca_cols = [f'PC_{i+1}' for i in range(X_train_extracted.shape[1])]
    pd.DataFrame(X_train_extracted, columns=feature_cols).to_csv('artifacts/X_train.csv', index=False)
    pd.DataFrame(X_test_extracted, columns=feature_cols).to_csv('artifacts/X_test.csv', index=False)
    pd.Series(y_train).to_csv('artifacts/y_train.csv', index=False)
    pd.Series(y_test).to_csv('artifacts/y_test.csv', index=False)
    
    # Export metrics metadata for frontend visualizations
    selection_metrics = {"features": feature_cols, "scores": feature_scores}#, "pca_variance": pca.explained_variance_ratio_.tolist()
    with open('artifacts/feature_analysis.json', 'w') as f:
        import json
        json.dump(selection_metrics, f)
        
    df.to_csv('artifacts/dashboard_data.csv', index=False)
    print("✅ Feature Extraction and Splitting tasks completed successfully without leakages!")

if __name__ == "__main__":
    prepare_select_and_extract()