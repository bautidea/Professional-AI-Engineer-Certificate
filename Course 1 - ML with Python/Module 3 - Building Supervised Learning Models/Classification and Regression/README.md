# 📊 Machine Learning with Python - Module 3: Classification and Regression

## 📖 Overview

In **Module 3 – Classification and Regression**, I explored various **supervised learning models** used to predict both **categorical** and **continuous values**. The focus was on:

- **Classification Models** → Assigning categorical labels to data points.
- **Regression Models** → Predicting continuous numerical values.
- **Decision Trees and Regression Trees** → Rule-based learning for classification and regression.
- **Understanding Tree-Based Models** → How they learn and make decisions.

This module introduced **structured decision-making algorithms**, improving upon simpler linear models, especially in handling **nonlinear relationships and hierarchical decision-making**.

---

## 📌 Topics Covered

### 1️⃣ **Classification: Categorizing Data into Groups**

Classification is a **supervised learning technique** where models categorize data points into **predefined labels**.

✔ **Binary Classification** → Two possible labels (e.g., spam vs. not spam).  
✔ **Multiclass Classification** → More than two categories (e.g., medical condition diagnosis).

#### **Multiclass Classification Strategies**

**1️⃣ One-vs-All (OvA) Strategy**

- Trains **K binary classifiers**, each distinguishing **one class vs. all others**.
- The class with the **highest probability score** is assigned.  
  ✔ **Simple and efficient**, especially for high-dimensional data.

**2️⃣ One-vs-One (OvO) Strategy**

- Trains **K(K-1)/2** classifiers, each distinguishing **two classes at a time**.
- A **voting mechanism** determines the final class label.  
  ✔ **More robust for similar classes**, but computationally expensive.

✔ **Use OvA when the dataset is large and computing power is limited.**  
✔ **Use OvO when classification accuracy is a priority over speed.**

---

### 2️⃣ **Decision Trees: Rule-Based Learning**

Decision Trees are **hierarchical models** used for both **classification and regression**. They **split** the dataset into smaller **homogeneous** groups by applying **feature-based decisions** recursively.

#### **How Decision Trees Work**

- **Root Node** → Represents the **entire dataset** and the **first splitting decision**.
- **Internal Nodes** → Represent **feature-based decision rules**.
- **Branches** → Represent the **outcome of decisions**.
- **Leaf Nodes** → Represent the **final class assignment (classification)** or **predicted value (regression)**.

Each split is **chosen to maximize the homogeneity** of the resulting subsets. The tree grows **until a stopping condition is met**.

#### **Splitting Criteria in Decision Trees**

1️⃣ **Entropy & Information Gain**  
✔ **Entropy** measures **uncertainty** in a node.  
✔ **Information Gain (IG)** measures how much entropy **decreases** after a split.

2️⃣ **Gini Impurity**  
✔ Measures **probability of incorrect classification**.  
✔ **Gini is computationally more efficient** than entropy.

✔ **Use Gini for speed, and Entropy for imbalanced datasets.**

#### **Preventing Overfitting: Pruning Decision Trees**

1️⃣ **Pre-Pruning (Early Stopping)** → Limits tree growth early.  
2️⃣ **Post-Pruning (Reduced Error Pruning)** → Removes unnecessary branches after training.

---

### 3️⃣ **Regression Trees: Predicting Continuous Values**

Regression Trees extend Decision Trees to **predict numerical values** instead of categories.

#### **How Regression Trees Work**

✔ The dataset is **split recursively** based on the **feature that minimizes prediction error**.  
✔ The final prediction at each leaf node is the **average of the target values**.

#### **Split Criteria for Regression Trees**

✔ **Mean Squared Error (MSE) Reduction** → The feature that results in the lowest variance in the target variable is chosen for splitting.  
✔ **Mean Absolute Error (MAE) Reduction** → Similar to MSE but less sensitive to outliers.

✔ **Regression Trees work well for nonlinear relationships but are sensitive to outliers.**

---

## **📂 Final Thoughts**

✅ **Classification Trees and Regression Trees provide structured, rule-based predictions.**  
✅ **Entropy, Gini, and MSE are used to evaluate splits and measure performance.**  
✅ **Pruning is crucial for improving decision tree generalization.**  
✅ **Scikit-learn provides easy-to-use implementations for Decision Trees and Regression Trees.**
