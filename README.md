# student_district_score_prediction
Classify the the probablity of a students of a particular district will perform either great or average nor poor using the existing dataset of class 5,6,8 
## Overview

This project predicts Class 10 learning outcome categories using National Achievement Survey (NAS) educational indicators from Class 3, Class 5, and Class 8.

The system applies multiple machine learning algorithms and uses XGBoost as the best-performing model for prediction.

---

## Features

- Interactive Streamlit Dashboard
- Real-Time Prediction
- XGBoost, Random Forest, SVM, KNN, Logistic Regression Models
- Feature Analysis Dashboard
- Model Comparison Visualization
- Feature Importance Visualization
- Confusion Matrix Analysis
- Prediction Confidence Scores

---

## Dataset

The dataset contains educational performance indicators collected from multiple districts and states.

Features include:

- Class 3 Language
- Class 3 Mathematics
- Class 3 EVS
- Class 5 Language
- Class 5 Mathematics
- Class 5 EVS
- Class 8 Language
- Class 8 Mathematics
- Class 8 Science
- Class 8 Social Science

Engineered Features:

- G3 Average
- G5 Average
- G8 Average
- Growth G3 to G5
- Growth G5 to G8
- Performance Standard Deviation

---

## Models Used

1. Logistic Regression
2. Support Vector Machine
3. K-Nearest Neighbors
4. Random Forest
5. XGBoost

---

## Best Performing Model

XGBoost Classifier

---

## Technologies Used

- Python
- Streamlit
- Scikit-Learn
- XGBoost
- Pandas
- NumPy
- Plotly

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/nas-learning-outcome-prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Project Structure

```text
app.py
requirements.txt
README.md
artifacts/
```
