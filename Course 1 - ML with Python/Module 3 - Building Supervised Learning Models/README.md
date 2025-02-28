# 📊 Machine Learning with Python - Module 3: Building Supervised Learning Models.

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

### 1️⃣ **Classification: Categorizing Data into Groups**

Classification is a **supervised learning technique** where models learn to categorize data points into **predefined labels**. The goal of classification is to identify which category a new data point belongs to based on patterns learned from labeled training data.

There are **two main types of classification problems**:

✔ **Binary Classification** → Two possible labels (e.g., spam vs. not spam).  
✔ **Multiclass Classification** → More than two categories (e.g., categorizing medical conditions).

Since many classification models are **binary by default**, special strategies are needed for **multiclass classification** (where there are more than two labels).

#### **1️⃣ One-vs-All (OvA) Strategy**

- Trains **K binary classifiers**, where each classifier learns to distinguish **one class vs. all others**.
- The final class label is assigned based on **which classifier assigns the highest probability score**.

✔ **Simple and efficient**, especially for high-dimensional data.  
✔ Works well when **one class is dominant over others**.

#### **2️⃣ One-vs-One (OvO) Strategy**

- Trains **K(K-1)/2** binary classifiers, each distinguishing between **two classes at a time**.
- A **voting mechanism** determines the final class label, based on which class wins the most pairwise comparisons.

✔ **More robust when classes are similar**, leading to **better decision boundaries**.  
✔ **Computationally expensive**, especially when there are **many classes**.

✔ **Use OvA when the dataset is large and computing power is limited.**  
✔ **Use OvO when classification accuracy is a priority over computation speed.**

---

### 2️⃣ **K-Nearest Neighbors (KNN): Instance-Based Learning**

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

#### **Choosing the Right K Value**

- **Small K (e.g., K=1)** → Leads to **overfitting**, as it considers only the closest point.
- **Large K (e.g., K=20)** → Leads to **underfitting**, as it averages too many neighbors.
- **Optimal K** → Found using cross-validation.

---

### 3️⃣ **Support Vector Machines (SVM): Hyperplane-Based Classification**

SVM is a **supervised learning technique** used for **classification and regression**. It is particularly useful for **high-dimensional spaces** where a clear **decision boundary** is required.

✔ **SVM finds the optimal hyperplane** that best separates two classes while maximizing the margin between them.

#### **How SVM Works**

1️⃣ **Maps data points into a higher-dimensional space**.  
2️⃣ **Finds the hyperplane that maximizes the margin between classes**.  
3️⃣ **Classifies new data points based on which side of the hyperplane they fall on**.

✔ **Works well for complex, non-linearly separable data using kernel tricks**.  
✔ **Sensitive to outliers**, which can influence the hyperplane.

#### **Types of SVM Kernels**

- **Linear Kernel:** Used when data is linearly separable.
- **Polynomial Kernel:** Captures curved decision boundaries.
- **Radial Basis Function (RBF) Kernel:** Projects data into a higher-dimensional space to create better separation.
- **Sigmoid Kernel:** Similar to neural network activation functions.

✔ **Use a linear kernel for simple datasets**.  
✔ **Use RBF when working with complex, non-linearly separable data**.

---

### 4️⃣ **Decision Trees: Rule-Based Learning**

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

### 5️⃣ **Regression Trees: Predicting Continuous Values**

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

### 6️⃣ **Bias-Variance Tradeoff**

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

### 7️⃣ **Ensemble Learning Techniques**

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
