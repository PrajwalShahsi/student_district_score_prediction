import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score
)
from xgboost import XGBClassifier


def train_and_export_models():

    print("🧠 Starting Step 4: Model Training Pipeline...")

    # Load prepared training data
    X_train = pd.read_csv('artifacts/X_train.csv')
    y_train = pd.read_csv('artifacts/y_train.csv').values.ravel()

    os.makedirs('artifacts', exist_ok=True)

    # =====================================================
    # SVM Hyperparameter Tuning
    # =====================================================

    print("\n🔍 Tuning SVM using GridSearchCV...")

    svm_param_grid = {
        'C': [0.1, 1, 10, 50, 100],
        'gamma': ['scale', 'auto', 0.01, 0.001],
        'kernel': ['rbf']
    }

    svm_grid = GridSearchCV(
        estimator=SVC(),
        param_grid=svm_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    svm_grid.fit(X_train, y_train)

    best_svm = svm_grid.best_estimator_

    print(f"\n✅ Best SVM Parameters: {svm_grid.best_params_}")
    print(f"✅ Best SVM CV Accuracy: {svm_grid.best_score_:.4f}")

    # =====================================================
    # Cross Validation of Best SVM
    # =====================================================

    cv_scores = cross_val_score(
        best_svm,
        X_train,
        y_train,
        cv=5,
        scoring='accuracy'
    )

    print(f"✅ 5-Fold CV Accuracy: {cv_scores.mean():.4f}")
    print("\n🔍 Tuning Random Forest...")

    rf_param_grid = {

        'n_estimators': [100, 200, 300, 500, 800],

        'max_depth': [5, 8, 10, 12, 15, 20, None],

        'min_samples_split': [2, 5, 10, 15],

        'min_samples_leaf': [1, 2, 4, 8],

        'max_features': ['sqrt', 'log2', None]
    }

    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=rf_param_grid,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    rf_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_

    print(f"\n✅ Best RF Parameters: {rf_search.best_params_}")
    print(f"✅ Best RF CV Accuracy: {rf_search.best_score_:.4f}")

    print("\n🔍 Tuning XGBoost...")

    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss'
        )

    xgb_param_grid = {
    'n_estimators': [300,500,800,1000],
    'max_depth': [3,4,5,6,7,8],
    'learning_rate': [0.01,0.03,0.05,0.1],
    'subsample': [0.7,0.8,0.9,1.0],
    'colsample_bytree': [0.7,0.8,0.9,1.0],
    'min_child_weight': [1,3,5],
    'gamma': [0,0.1,0.2,0.3]
    }

    xgb_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=xgb_param_grid,
        n_iter=150,
        cv=8,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    xgb_search.fit(X_train, y_train)

    best_xgb = xgb_search.best_estimator_

    print(f"\n✅ Best XGB Parameters: {xgb_search.best_params_}")
    print(f"✅ Best XGB CV Accuracy: {xgb_search.best_score_:.4f}")
    # =====================================================
    # Models
    # =====================================================

    models_to_train = {

        'random_forest_model.pkl': best_rf,

        'logistic_regression_model.pkl': LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        'svm_model.pkl': best_svm,

        'knn_model.pkl': KNeighborsClassifier(
            n_neighbors=7
        ),
        'xgboost_model.pkl': best_xgb,
        
    }
    
    # =====================================================
    # Train & Save Models
    # =====================================================

    for filename, model_object in models_to_train.items():

        print(f"\n🚀 Training {filename}...")

        model_object.fit(X_train, y_train)

        joblib.dump(
            model_object,
            os.path.join('artifacts', filename)
        )

        print(f"✅ Saved: artifacts/{filename}")

    print("\n🎉 All models trained and saved successfully!")


if __name__ == "__main__":
    train_and_export_models()