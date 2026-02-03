# 🏥 Advanced Deep Learning for Disease Risk Prediction

## Project Overview
Early identification of disease risk is a key challenge in preventive healthcare. This project implements an advanced and interpretable **Disease Risk Prediction System** using **Machine Learning (ML)** and **Deep Learning (DL)** techniques on structured health and lifestyle data.

A complete data science pipeline is followed, including exploratory data analysis, preprocessing, feature engineering, feature selection, model comparison, hyperparameter tuning, explainable AI (XAI), and deep learning modeling.

The goal of the project is not only to achieve predictive accuracy but also to ensure **model transparency, robustness, and responsible use** in healthcare-related applications.

---

## Problem Statement
The objective of this project is to classify individuals into **low-risk** or **high-risk** disease categories based on demographic, lifestyle, and physiological health indicators.

- **Problem Type:** Binary Classification  
- **Target Variable:** Disease Risk (0 = Low Risk, 1 = High Risk)

---

## Dataset Description
- **Dataset:** Health & Lifestyle Dataset  
- **Type:** Structured tabular data  
- **Features Include:**
  - Demographic factors (Age)
  - Lifestyle habits (Physical activity, sleep patterns, diet)
  - Physiological indicators (BMI, blood pressure, cholesterol)
  - Family medical history

The dataset is clean, balanced, and suitable for supervised learning.

---

## Project Pipeline
1. Exploratory Data Analysis (EDA)  
2. Data Preprocessing & Standardization  
3. Feature Engineering  
4. Feature Selection  
5. Machine Learning Model Training  
6. Hyperparameter Tuning  
7. Explainable AI (SHAP & Feature Importance)  
8. Deep Learning Model (ANN)  
9. Critical Analysis & Conclusion  

---

## Models Implemented
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- Artificial Neural Network (ANN)  

---

## Explainability (XAI)
Explainable AI techniques were applied to improve model transparency:
- SHAP summary plots for global feature importance  
- SHAP local explanations for individual predictions  
- Random Forest feature importance analysis  

These techniques help identify the key factors influencing disease risk predictions.

---

## Deep Learning Model
An **Artificial Neural Network (ANN)** was developed for tabular data:
- Fully connected dense layers with ReLU activation  
- Dropout regularization to reduce overfitting  
- Binary sigmoid output layer  
- Loss curve analysis to monitor training and validation performance  

The ANN achieved competitive results but did not outperform the best classical ML models.

---

## Model Performance (Accuracy)
| Model | Accuracy |
|------|----------|
| Logistic Regression | 0.7518 |
| KNN | 0.6978 |
| Decision Tree | 0.6152 |
| Random Forest | 0.7517 |
| Linear SVM | 0.7518 |
| ANN | 0.7518 |
---

## Key Insights
- Tree-based models perform effectively on structured health data  
- Feature engineering significantly improves performance  
- ANN provides competitive results with higher complexity  
- SHAP explainability enhances trust and interpretability  
- Model transparency is critical in healthcare applications  

---

## Limitations
- Predictions indicate **risk**, not medical diagnosis  
- Dataset does not capture temporal health variations  
- Generalization may vary across populations  
- Deep learning interpretability remains limited  

---

## Future Work
- Incorporate longitudinal health data  
- Integrate wearable or sensor-based information  
- Apply advanced ensemble or hybrid models  
- Deploy as a clinical decision-support system  

---

## Project Structure

├── Data/
│ └── health_lifestyle_dataset.csv
├── Disease_Risk_Prediction_ML_DL.ipynb
├── README.md
├── Disease_Risk_Prediction_Report.pdf

---

## How to Run
1. Open `Disease_Risk_Prediction_ML_DL.ipynb` in Jupyter Notebook or Google Colab  
2. Install required libraries (scikit-learn, pandas, numpy, tensorflow, shap)  
3. Run all cells sequentially  

---
