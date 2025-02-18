# Machine Learning in Action - Module 2: Logistic Regression

## Overview

In this module, I explored **Logistic Regression**, a fundamental classification algorithm in Machine Learning. Unlike Linear Regression, which predicts continuous values, Logistic Regression estimates **probabilities** and maps them to binary classes (0 or 1).

Throughout this module, I learned how to apply **Logistic Regression**, interpret its outputs, optimize its parameters, and evaluate its performance using **log-loss, accuracy, precision, recall, and F1-score**. Additionally, I explored **Gradient Descent and Stochastic Gradient Descent (SGD)** as optimization techniques.

---

## **Key Topics Covered**

### 1️⃣ **Introduction to Logistic Regression**

- **Definition:** Logistic Regression is a **binary classification** algorithm that predicts the probability of an outcome belonging to one of two classes.
- **When to Use It:**

  - When the **target variable** is binary (e.g., spam vs. non-spam).
  - When I need **probabilistic outputs** (e.g., probability of customer churn).
  - When I need to assess the **impact of an independent feature** on the outcome.

- **Mathematical Representation:**

  - Logistic Regression transforms a linear function using the **sigmoid function**, which maps predictions between 0 and 1:
    **p(ŷ = 1 | x) = 1 / (1 + e^-(θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ))**
  - The **decision boundary** is determined by setting a threshold (commonly **0.5**).

- **Why the Sigmoid Function?**
  - It forces outputs to be between **0 and 1**.
  - It provides **a smooth, continuous curve** instead of abrupt step changes.
  - It enables **probabilistic interpretation** of predictions.

---

### 2️⃣ **Training a Logistic Regression Model**

- **Objective:** Find the best parameters (θ) that map **input features to class probabilities** while minimizing errors.
- **Key Steps in Model Training:**

  1. Initialize **random parameters** (θ₀, θ₁, ..., θₙ).
  2. Compute the **predicted probability** for each observation.
  3. Measure **error** using the **log-loss function**.
  4. Update the parameters using an **optimization algorithm**.
  5. Repeat until convergence or until a **maximum number of iterations** is reached.

- **Optimization Techniques:**

  - **Gradient Descent:**

    - Adjusts **θ values** iteratively to minimize **log-loss**.
    - Moves in the **direction of steepest descent** (negative gradient).
    - Requires tuning of the **learning rate** (step size).

  - **Stochastic Gradient Descent (SGD):**
    - Uses a **random subset of the dataset** instead of the entire dataset.
    - Converges **faster** than standard Gradient Descent.
    - Introduces some **randomness**, which can help escape local minima.

- **Log-Loss Function:**
  - Measures how well predicted probabilities match the actual class labels.
  - Penalizes **incorrect confident predictions** more than mild errors.
  - The **goal** is to minimize log-loss to improve classification performance.

---

### 3️⃣ **Evaluating Logistic Regression Performance**

- **Why Evaluation Matters:**

  - Since Logistic Regression outputs probabilities, I must **convert them into class labels** using a threshold.
  - The choice of **threshold impacts model performance**, and different problems require different thresholds.

- **Key Evaluation Metrics:**

  - **Accuracy:** Measures the percentage of correct predictions.
  - **Precision:** Fraction of positive predictions that were actually correct.
  - **Recall (Sensitivity):** Measures how well the model identifies true positives.
  - **F1-Score:** Harmonic mean of precision and recall (useful for imbalanced datasets).
  - **ROC Curve & AUC Score:** Visualizes how well the model distinguishes between classes.

- **Choosing the Right Metric:**
  - If **false positives** are costly (e.g., fraud detection), **precision** is more important.
  - If **false negatives** are costly (e.g., medical diagnosis), **recall** is more critical.
  - If I need a **balance**, the **F1-score** is a good metric.

---

## **Insights from Notebooks**

### **Logistic Regression Model Training**

- Explored how **changing the learning rate** affects model convergence.
- Compared **Gradient Descent vs. Stochastic Gradient Descent** and analyzed trade-offs.
- Evaluated the impact of **feature scaling** on performance and optimization speed.
- Learned that **regularization** (L1/L2) helps prevent **overfitting**.

---

## **Key Takeaways**

✅ **Logistic Regression is a powerful yet simple model for binary classification.**  
✅ **The sigmoid function converts linear outputs into probabilities between 0 and 1.**  
✅ **Model optimization relies on minimizing log-loss using Gradient Descent or SGD.**  
✅ **Evaluation metrics like precision, recall, and F1-score help assess model effectiveness.**  
✅ **The choice of decision threshold directly impacts classification performance.**

---
