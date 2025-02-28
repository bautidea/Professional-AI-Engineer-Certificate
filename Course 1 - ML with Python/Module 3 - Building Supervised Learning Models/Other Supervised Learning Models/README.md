# 📊 Machine Learning with Python - Module 3: Other Supervised Learning Models

## 📖 Overview

In **Module 3 – Other Supervised Learning Models**, I explored additional supervised learning techniques beyond decision trees. The focus was on:

- **K-Nearest Neighbors (KNN)** for instance-based classification.
- **Support Vector Machines (SVM)** for optimal decision boundary determination.
- **The Bias-Variance Tradeoff**, its impact on model performance, and techniques to mitigate it.
- **Ensemble Learning**, including **Bagging (Random Forests)** and **Boosting (XGBoost, AdaBoost)** to improve prediction accuracy.

This module introduced **more complex decision-making algorithms**, improving upon simpler linear models, especially in handling **nonlinear relationships and high-dimensional data**.

---

## 📌 Topics Covered

### 1️⃣ **K-Nearest Neighbors (KNN): Instance-Based Learning**

K-Nearest Neighbors (**KNN**) is a **non-parametric**, **instance-based learning algorithm** used for **both classification and regression**.

✔ **It does not explicitly learn a model during training**. Instead, it stores all training data and makes predictions **based on similarity**.

#### **How KNN Works**

1️⃣ **Choose a value for K** (the number of neighbors).  
2️⃣ **Calculate the distance** between the query point and all training data points.  
3️⃣ **Find the K-nearest points** to the query point.  
4️⃣ **For classification:** Assign the most common class among the K neighbors.  
5️⃣ **For regression:** Compute the average (or weighted average) of the K-nearest neighbors’ values.

✔ **KNN is effective for small datasets with low-dimensional data**.  
✔ **Sensitive to noisy data** if K is too low.

---

### 2️⃣ **Support Vector Machines (SVM): Hyperplane-Based Classification**

SVM is a **supervised learning technique** used for **classification and regression**. It is particularly useful for **high-dimensional spaces** where a clear **decision boundary** is required.

✔ **SVM finds the optimal hyperplane** that best separates two classes while maximizing the margin between them.

#### **Types of SVM Kernels**

- **Linear Kernel:** Used when data is linearly separable.
- **Polynomial Kernel:** Captures curved decision boundaries.
- **Radial Basis Function (RBF) Kernel:** Projects data into a higher-dimensional space to create better separation.
- **Sigmoid Kernel:** Similar to neural network activation functions.

✔ **Use a linear kernel for simple datasets**.  
✔ **Use RBF when working with complex, non-linearly separable data**.

---

### 3️⃣ **Bias-Variance Tradeoff**

The **Bias-Variance Tradeoff** is a fundamental concept in machine learning that describes how a model's complexity impacts its ability to **generalize to new data**. The goal of machine learning is to build a model that can **accurately predict outcomes on unseen data**, not just memorize training examples.

#### **What is Bias?**

Bias refers to **assumptions** a model makes about the relationship between input features and target labels. It is a measure of how far the model’s **predicted values** deviate from the actual values.

- **High Bias (Underfitting)**
  - The model is **too simple** and **fails to capture the complexity** of the data.
  - It makes **strong assumptions**, ignoring relationships between features.
  - **Consequences**:
    - The model performs **poorly on both training and test data**.
    - Produces **similar predictions** regardless of input variation.

✔ **Low bias leads to more accurate predictions**.  
✖ **High bias leads to an oversimplified model that misses key patterns**.

#### **What is Variance?**

Variance refers to **how much a model’s predictions change** when trained on **different subsets of data**. A model with **high variance** is overly **sensitive to training data** and captures **random noise** instead of the underlying pattern.

- **High Variance (Overfitting)**
  - The model is **too complex** and **memorizes** the training data.
  - It captures **irrelevant noise**, rather than learning the actual pattern.
  - **Consequences**:
    - Performs **very well on training data** but **poorly on test data**.

✔ **Low variance means the model is stable across different training datasets**.  
✖ **High variance causes models to be inconsistent and unreliable on new data**.

---

### **The Tradeoff: Balancing Bias and Variance**

| **Model Complexity**                   | **Bias** | **Variance** | **Performance on Training Data** | **Performance on Test Data** |
| -------------------------------------- | -------- | ------------ | -------------------------------- | ---------------------------- |
| **Simple Model (Underfitting)**        | 🔴 High  | ✅ Low       | ❌ Poor                          | ❌ Poor                      |
| **Balanced Model (Good Fit)**          | ✅ Low   | ✅ Low       | ✅ Good                          | ✅ Good                      |
| **Overly Complex Model (Overfitting)** | ✅ Low   | 🔴 High      | ✅ Excellent                     | ❌ Poor                      |

✔ **Goal:** Find the **optimal model complexity** where the model is **complex enough to learn from data** but **not too complex to memorize noise**.

---

### 4️⃣ **Ensemble Learning Techniques**

**Ensemble Learning** combines multiple models to create a more **robust, stable, and accurate** prediction system. Instead of relying on a **single model**, ensemble methods **combine the predictions of multiple models** to produce a **stronger final result**.

#### **Bagging (Bootstrap Aggregating)**

- **Definition:** Bagging is an ensemble technique that **reduces variance** by training multiple **independent models** on different **random subsets of data**.
- **How it Works:**
  1️⃣ Multiple copies of the training dataset are created using **random sampling with replacement (bootstrap sampling)**.  
  2️⃣ A separate model (e.g., Decision Tree) is trained on each subset.  
  3️⃣ Predictions from all models are **averaged (for regression)** or **majority-voted (for classification)**.

- **Example:** **Random Forests** (an ensemble of multiple decision trees using bagging).
- **Advantages:**  
  ✔ Reduces **overfitting** by stabilizing predictions.  
  ✔ Works well with **high-variance models** (like deep decision trees).  
  ✔ Improves **model accuracy** by averaging multiple weak learners.

---

#### **Boosting**

- **Definition:** Boosting is an ensemble technique that **reduces bias** by **sequentially training weak models**, where each model **corrects the errors** of the previous model.
- **How it Works:**
  1️⃣ The first weak model is trained on the dataset.  
  2️⃣ The next model **focuses on examples misclassified by the previous model**.  
  3️⃣ This process repeats, progressively improving accuracy.  
  4️⃣ The final prediction is **a weighted sum** of all weak models.

- **Examples:**

  - **XGBoost** (Extreme Gradient Boosting)
  - **AdaBoost** (Adaptive Boosting)
  - **Gradient Boosting**

- **Advantages:**  
  ✔ Reduces **bias**, making weak models stronger.  
  ✔ Works well with **imbalanced datasets**.  
  ✔ Often achieves **higher accuracy** than bagging methods.

---

#### **Random Forest vs. XGBoost**

| **Method**        | **Works By**                                                  | **Strengths**                               | **Weaknesses**               |
| ----------------- | ------------------------------------------------------------- | ------------------------------------------- | ---------------------------- |
| **Random Forest** | Uses **Bagging** to train multiple independent decision trees | Reduces **variance**, prevents overfitting  | Slower inference             |
| **XGBoost**       | Uses **Boosting** to sequentially correct weak learners       | Reduces **bias**, handles complex data well | Sensitive to hyperparameters |

✔ **Bagging is used when variance is high (to stabilize predictions).**  
✔ **Boosting is used when bias is high (to improve weak models).**  
✔ **Both methods improve generalization performance.**

---
