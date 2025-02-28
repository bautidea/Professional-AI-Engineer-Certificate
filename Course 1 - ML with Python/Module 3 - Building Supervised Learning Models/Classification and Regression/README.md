# 📊 Machine Learning with Python - Module 3: Classification and Regression

## 📖 Overview

In **Module 3**, I explored various **supervised learning models** used for both **classification** and **regression** tasks. The focus was on:

- **Decision Trees** and **Regression Trees** for structured rule-based learning.
- **K-Nearest Neighbors (KNN)** for instance-based classification.
- **Support Vector Machines (SVM)** for optimal decision boundary determination.
- **The Bias-Variance Tradeoff**, its impact on model performance, and techniques to mitigate it.
- **Ensemble Learning**, including **Bagging (Random Forests)** and **Boosting (XGBoost, AdaBoost)** to improve prediction accuracy.

This module introduced **more complex decision-making algorithms** that improve upon simpler linear models, especially in handling **nonlinear relationships, structured decisions, and high-dimensional data**.

---

## 📌 Topics Covered

### 1️⃣ **Decision Trees: Rule-Based Learning**

Decision Trees are **hierarchical models** used for both **classification and regression**. They **split** the dataset into smaller **homogeneous** groups by applying **feature-based decisions** recursively.

### **How Decision Trees Work**

A Decision Tree is a **tree-like structure** where:

- **Root Node** → Represents the **entire dataset** and the **first splitting decision**.
- **Internal Nodes** → Represent **feature-based decision rules**.
- **Branches** → Represent the **outcome of decisions**.
- **Leaf Nodes** → Represent the **final class assignment (classification)** or **predicted value (regression)**.

Each split is **chosen to maximize the homogeneity** of the resulting subsets. The tree continues to grow **until a stopping condition is met**, such as:

- A **predefined maximum depth**.
- The **minimum number of samples per node** is reached.
- The **node is already pure** (i.e., all samples belong to the same class).

---

### **Splitting Criteria in Decision Trees**

The **quality of a split** is determined by measuring how **homogeneous** the resulting subsets are. The **most important splitting criteria** are:

1️⃣ **Entropy & Information Gain**

- **Entropy** measures **uncertainty** in a node. A **pure node** (containing only one class) has entropy **0**, while a node with **equal class distribution** has **maximum entropy**.
- **Information Gain (IG)** calculates how much entropy **decreases** after a split. The **feature with the highest IG is chosen** as the split.

2️⃣ **Gini Impurity**

- Measures the **probability of incorrect classification**.
- Like Entropy, **a Gini Impurity of 0** means the node is **completely pure**.
- **Gini is computationally more efficient** than entropy.

**Choosing between Gini and Entropy:**
✔ **Gini is faster** (less computation).  
✔ **Entropy is useful** when dealing with **imbalanced datasets**.

---

### **Pruning: Preventing Overfitting in Decision Trees**

**Overfitting occurs** when the tree **memorizes training data** instead of learning generalizable patterns.  
**Pruning techniques** help **simplify trees** and improve **generalization**:

1️⃣ **Pre-Pruning (Early Stopping)**

- **Limits tree depth** to prevent overly complex models.
- **Stops splitting when samples in a node are too few**.

2️⃣ **Post-Pruning (Reduced Error Pruning)**

- **Removes branches that do not improve accuracy**.
- **Uses cross-validation** to prune overfitted branches.

---

### 2️⃣ **Regression Trees: Predicting Continuous Values**

Regression Trees extend Decision Trees to **predict numerical values** instead of categories. Instead of classifying samples, they estimate a **continuous target variable** by recursively splitting the dataset.

### **How Regression Trees Work**

1️⃣ **Find the best feature and threshold to split the data.**  
2️⃣ **At each split, the goal is to minimize prediction error.**  
3️⃣ **The final prediction at each leaf node is the average of the target values.**

### **Split Criteria for Regression Trees**

Unlike classification, where we use **Gini Impurity or Information Gain**, regression trees use:

- **Mean Squared Error (MSE) Reduction**: The feature that results in the **lowest variance** in the target variable is chosen for splitting.
- **Mean Absolute Error (MAE) Reduction**: Similar to MSE but less sensitive to outliers.

✔ **Advantages:** Works well for **nonlinear relationships**.  
✖ **Limitations:** Sensitive to **outliers and overfitting**.

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
