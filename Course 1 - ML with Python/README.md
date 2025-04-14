# 🤖 Machine Learning with Python – Course 1 (IBM ML Certificate)

This repo contains all the work I did for **Course 1 – Machine Learning with Python**, part of the [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning). It goes from foundational supervised models like linear regression to unsupervised learning, evaluation, and best practices.

---

## 📚 Course Content Overview

The course is organized into 5 modules + 1 final project. Each module has its own folder with labs, transcripts, and a full summary (`README.md`) with detailed insights and notes. Here's what was covered:

---

## 📦 Module 1 – Introduction to Machine Learning

- What is Machine Learning? (Supervised, Unsupervised, Reinforcement)
- Common use cases across industries.
- Overview of ML pipelines and where ML fits in the data science process.
- No coding yet—just core ML understanding and terminology.

---

## 📈 Module 2 – Linear & Logistic Regression

**Topics:**

- Simple Linear Regression (SLR)
- Multiple Linear Regression (MLR)
- Logistic Regression (binary classification)

**Key Concepts:**

- OLS (Ordinary Least Squares) to fit regression models.
- MSE as a loss function.
- The sigmoid function for logistic regression.
- Learned to model relationships between features and targets (numeric or binary).
- Evaluated regression fits and prediction accuracy.

---

## 🌲 Module 3 – Classification & Regression Models

Split into two parts:

### 🔹 Part 1: Decision & Regression Trees

- Decision Trees for classification using entropy, information gain, and Gini impurity.
- Regression Trees for predicting continuous values using MSE-based split criteria.
- Tree pruning to avoid overfitting.

### 🔹 Part 2: KNN, SVM & Ensemble Methods

- **K-Nearest Neighbors (KNN)**: instance-based learner, works on proximity.
- **Support Vector Machines (SVM)**: uses optimal separating hyperplanes.
- **Ensemble Learning**:
  - Bagging (Random Forests) → reduces variance.
  - Boosting (XGBoost, AdaBoost) → reduces bias.
- **Bias-Variance Tradeoff** and how ensemble models help with generalization.

---

## 🧪 Module 4 – Unsupervised Learning

### 🔸 Clustering

- K-Means clustering (centroid-based)
- DBSCAN & HDBSCAN (density-based)
- Hierarchical clustering (agglomerative/divisive)
- Use cases: customer segmentation, anomaly detection, image segmentation.

### 🔸 Dimensionality Reduction & Feature Engineering

- PCA (linear reduction)
- t-SNE and UMAP (nonlinear, good for visualization)
- Explained how these techniques reduce dimensions while keeping relevant structure.
- Connected clustering + dimensionality reduction + feature selection.

---

## ✅ Module 5 – Evaluating and Validating ML Models

### 🔹 Part 1: Evaluation Techniques

- **Classification Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- **Regression Metrics**: MAE, MSE, RMSE, R², Residual plots
- **Unsupervised Evaluation**:
  - Internal: Silhouette score, Davies-Bouldin, Calinski-Harabasz
  - External: Adjusted Rand Index, Normalized Mutual Info
  - Dimensionality evaluation: Explained variance, reconstruction error

### 🔹 Part 2: Best Practices for Generalization

- Train/Validation/Test splits
- K-Fold and Stratified Cross-Validation
- TimeSeriesSplit for temporal datasets
- **Regularization**:
  - Ridge (L2): shrinks all coefficients
  - Lasso (L1): shrinks some to zero (feature selection)
- **Avoiding Pitfalls**:
  - Data leakage & data snooping
  - Misinterpreting feature importance
  - Causal inference mistakes

---

## 🎯 Final Project – Rain Prediction in Australia

- **Goal**: Predict `RainTomorrow` using weather data.
- Applied preprocessing, visualization, multiple models (Logistic Regression, Random Forest).
- Used proper train/test split, cross-validation, hyperparameter tuning, evaluation.
- Avoided data leakage and addressed class imbalance.

---

## 🛠 Tools & Libraries

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- XGBoost, HDBSCAN
- PCA, t-SNE, UMAP

---
