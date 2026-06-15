
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib, json, os
from PIL import Image

st.set_page_config(page_title="NAS Learning Outcome Predictor", page_icon="🎓", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

@st.cache_resource
def load_models():
    models = {}
    files = {
        "Best Model (XGBoost)": "best_model.pkl",
        "XGBoost": "xgboost_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "Logistic Regression": "logistic_regression_model.pkl",
        "SVM": "svm_model.pkl",
        "KNN": "knn_model.pkl"
    }
    for k,v in files.items():
        path = os.path.join(ARTIFACTS,v)
        if os.path.exists(path):
            models[k] = joblib.load(path)

    return {
        "models": models,
        "scaler": joblib.load(os.path.join(ARTIFACTS,"scaler.pkl")),
        "le_state": joblib.load(os.path.join(ARTIFACTS,"le_state.pkl")),
        "le_target": joblib.load(os.path.join(ARTIFACTS,"le_target.pkl"))
    }

@st.cache_data
def load_data():
    dashboard = pd.read_csv(os.path.join(ARTIFACTS,"dashboard_data.csv"))
    engineered = pd.read_csv(os.path.join(ARTIFACTS,"engineered_features.csv"))
    comparison = pd.read_csv(os.path.join(ARTIFACTS,"model_comparison.csv"))

    with open(os.path.join(ARTIFACTS,"feature_analysis.json")) as f:
        fa = json.load(f)

    with open(os.path.join(ARTIFACTS,"metrics.json")) as f:
        metrics = json.load(f)

    return dashboard, engineered, comparison, fa, metrics

assets = load_models()
dashboard, engineered, comparison, fa, metrics = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home","📊 Analytics","🤖 Prediction"]
)

if page == "🏠 Home":
    st.title("🎓 NAS District Learning Outcome Prediction System")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Districts", len(engineered))
    c2.metric("States/UTs", engineered["state_name"].nunique())
    c3.metric("Models", 6)
    c4.metric("Best Model", "XGBoost")

    st.markdown("""
    ### Project Objective
    Predict Class 10 learning outcome category using NAS Class 3, 5 and 8 performance indicators.
    This dashboard includes model evaluation, feature analysis and real-time prediction.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(engineered.head(), use_container_width=True)

elif page == "📊 Analytics":
    st.title("📊 Model Analytics Dashboard")

    tab1,tab2,tab3,tab4 = st.tabs(
        ["Feature Scores","Model Comparison","Feature Importance","Confusion Matrix"]
    )

    with tab1:
        feature_df = pd.DataFrame({
            "Feature": fa["features"],
            "Score": fa["scores"]
        }).sort_values("Score", ascending=False)

        fig = px.bar(feature_df,
                     x="Score",
                     y="Feature",
                     orientation="h",
                     title="ANOVA Feature Ranking")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.dataframe(comparison, use_container_width=True)

        fig = px.bar(comparison,
                     x=comparison.columns[0],
                     y=comparison.columns[1],
                     title="Model Accuracy Comparison")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        img_path = os.path.join(ARTIFACTS,"feature_importance.png")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), use_container_width=True)

    with tab4:
        img_path = os.path.join(ARTIFACTS,"confusion_matrix.png")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), use_container_width=True)

elif page == "🤖 Prediction":
    st.title("🤖 Live Prediction Engine")

    model_name = st.selectbox(
        "Choose Model",
        list(assets["models"].keys())
    )

    state = st.selectbox(
        "State",
        sorted(assets["le_state"].classes_)
    )

    col1,col2,col3 = st.columns(3)

    with col1:
        g3_lang = st.slider("Class 3 Language",0,100,50)
        g3_math = st.slider("Class 3 Math",0,100,50)
        g3_evs = st.slider("Class 3 EVS",0,100,50)

    with col2:
        g5_lang = st.slider("Class 5 Language",0,100,50)
        g5_math = st.slider("Class 5 Math",0,100,50)
        g5_evs = st.slider("Class 5 EVS",0,100,50)

    with col3:
        g8_lang = st.slider("Class 8 Language",0,100,50)
        g8_math = st.slider("Class 8 Math",0,100,50)
        g8_sci = st.slider("Class 8 Science",0,100,50)
        g8_sst = st.slider("Class 8 Social Science",0,100,50)

    if st.button("Predict"):
        g3_avg = np.mean([g3_lang,g3_math,g3_evs])
        g5_avg = np.mean([g5_lang,g5_math,g5_evs])
        g8_avg = np.mean([g8_lang,g8_math,g8_sci,g8_sst])

        row = [[
            g3_lang,g3_math,g3_evs,
            g5_lang,g5_math,g5_evs,
            g8_lang,g8_math,g8_sci,g8_sst,
            g3_avg,g5_avg,g8_avg,
            g5_avg-g3_avg,
            g8_avg-g5_avg,
            np.std([g3_lang,g3_math,g3_evs,g5_lang,g5_math,g5_evs,g8_lang,g8_math,g8_sci,g8_sst]),
            assets["le_state"].transform([state])[0]
        ]]

        scaled = assets["scaler"].transform(row)

        model = assets["models"][model_name]

        pred = model.predict(scaled)
        label = assets["le_target"].inverse_transform(pred)[0]

        st.success(f"Predicted Category: {label}")

        if hasattr(model,"predict_proba"):
            probs = model.predict_proba(scaled)[0]
            classes = assets["le_target"].classes_

            prob_df = pd.DataFrame({
                "Category": classes,
                "Probability": probs
            })

            fig = px.bar(prob_df,
                         x="Category",
                         y="Probability",
                         title="Prediction Confidence")
            st.plotly_chart(fig, use_container_width=True)
