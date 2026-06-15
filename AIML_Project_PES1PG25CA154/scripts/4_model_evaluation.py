import pandas as pd
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def evaluate_model():
    
    X_test = pd.read_csv('artifacts/X_test.csv')
    y_test = pd.read_csv('artifacts/y_test.csv').values.ravel()

    le_target = joblib.load('artifacts/le_target.pkl')

    models = {
        "Random Forest": joblib.load('artifacts/random_forest_model.pkl'),
        "Logistic Regression": joblib.load('artifacts/logistic_regression_model.pkl'),
        "SVM": joblib.load('artifacts/svm_model.pkl'),
        "KNN": joblib.load('artifacts/knn_model.pkl'),
        "XGBoost": joblib.load('artifacts/xgboost_model.pkl')
    }

    print("\n========== MODEL COMPARISON ==========")

    best_accuracy = 0
    best_model = None
    best_model_name = None
    best_metrics = None
    model_scores = {}
    for model_name, model in models.items():

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        model_scores[model_name] = accuracy
        print(f"{model_name}: {accuracy:.4f}")

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

            best_metrics = {

                "accuracy": accuracy,

                "classification_report":
                    classification_report(
                        y_test,
                        y_pred,
                        target_names=le_target.classes_,
                        output_dict=True
                    ),

                "confusion_matrix":
                    confusion_matrix(
                        y_test,
                        y_pred
                    ).tolist(),

                "feature_importances":
                    model.feature_importances_.tolist()
                    if hasattr(model, 'feature_importances_')
                    else [],

                "features": list(X_test.columns),

                "classes": list(le_target.classes_)
            }

    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"🏆 Best Accuracy: {best_accuracy:.4f}")
    xgb = joblib.load('artifacts/xgboost_model.pkl')

    importances = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': xgb.feature_importances_
    })

    print(
         importances.sort_values(
         by='Importance',
         ascending=False
        )
    )
    joblib.dump(
        best_model,
        'artifacts/best_model.pkl'
    )

    with open('artifacts/metrics.json', 'w') as f:
        json.dump(best_metrics, f)

    print("✅ best_model.pkl updated")
    print("✅ metrics.json saved")
    comparison_df = pd.DataFrame(
     model_scores.items(),
        columns=["Model", "Accuracy"]
    )

    comparison_df.to_csv(
     "artifacts/model_comparison.csv",
     index=False
    )

    print("✅ model_comparison.csv saved")

    import matplotlib.pyplot as plt

    importance_df = pd.DataFrame({
    "Feature": best_metrics["features"],
    "Importance": best_metrics["feature_importances"]
    })

    importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
    )

    plt.figure(figsize=(10,6))
    plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
    )

    plt.tight_layout()

    plt.savefig(
    "artifacts/feature_importance.png"
    )
    

    print("✅ feature_importance.png saved")

    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=le_target.classes_,
    yticklabels=le_target.classes_
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
    "artifacts/confusion_matrix.png"
    )

    print("✅ confusion_matrix.png saved")

    
if __name__ == "__main__":
    evaluate_model()