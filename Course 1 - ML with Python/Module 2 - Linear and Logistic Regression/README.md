# Machine Learning in Action - Module 2: Regression (Linear & Logistic)

## Overview

Module 2 explores **Regression** as a supervised learning technique, covering **Linear Regression (Simple & Multiple)** and **Logistic Regression**. The module focuses on **predicting continuous and categorical outcomes**, understanding model assumptions, and evaluating performance using key metrics.

## Key Topics Covered

### 1️⃣ **Linear Regression**

- **Definition:** Models the relationship between a continuous target variable and one or more explanatory features.
- **Types:**
  - **Simple Linear Regression:** Uses a **single** independent variable.
  - **Multiple Linear Regression:** Extends to **multiple** independent variables.
- **Mathematical Representation:**
  - **Simple Linear Regression:**  
    **ŷ = θ₀ + θ₁x**
  - **Multiple Linear Regression:**  
    **ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ**
- **Model Assumptions:**

  - **Linearity:** Relationship between predictors and target is linear.
  - **Independence:** Features should not be highly correlated (multicollinearity).
  - **Homoscedasticity:** Variance remains constant across predictions.
  - **Normality of Residuals:** Error terms should be normally distributed.

- **Evaluation Metrics:**

  - **Mean Squared Error (MSE):** Measures average squared differences between actual and predicted values.
  - **R-squared (R²):** Proportion of variance explained by the model.
  - **Root Mean Squared Error (RMSE):** Square root of MSE for easier interpretation.

- **Challenges & Considerations:**
  - **Overfitting:** Using too many features can cause the model to memorize the training data rather than generalizing.
  - **Outliers:** High-leverage points can disproportionately influence the regression line.
  - **Feature Scaling:** Standardization improves model convergence and accuracy.

---

### 2️⃣ **Polynomial & Non-Linear Regression**

- **Why Use Non-Linear Regression?**

  - When data does not fit a straight-line relationship.
  - Captures more complex relationships using **higher-degree polynomials, logarithmic, or exponential functions**.

- **Polynomial Regression:**

  - Extends linear regression by adding polynomial terms:
    **ŷ = θ₀ + θ₁x + θ₂x² + θ₃x³ + ... + θₙxⁿ**
  - **Caution:** Higher-degree polynomials can cause **overfitting**.

- **Other Non-Linear Models:**
  - **Exponential Models:** Used for growth-based relationships.
  - **Logarithmic Models:** Suitable for diminishing returns or saturation effects.

---

### 3️⃣ **Logistic Regression**

- **Definition:** A classification algorithm that predicts the **probability of an outcome belonging to one of two classes**.
- **Mathematical Representation:**

  - Uses the **sigmoid function** to convert predictions into probabilities:
    **p(ŷ = 1 | x) = 1 / (1 + e^-(θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ))**
  - **Decision Boundary:** Classification is determined by a threshold (commonly **0.5**).

- **Evaluation Metrics:**

  - **Log-Loss (Cross-Entropy):** Measures how well predicted probabilities match actual labels.
  - **Accuracy, Precision, Recall, F1-score:** Assess classification performance.

- **Optimization Techniques:**
  - **Gradient Descent:** Iteratively updates parameters to minimize log-loss.
  - **Stochastic Gradient Descent (SGD):** Uses random subsets of data for faster convergence.

---

## **Insights from Notebooks**

### **Simple Linear Regression Notebook**

- Demonstrates **best-fit line** estimation using **Ordinary Least Squares (OLS)**.
- Analyzes the impact of **feature scaling and residual analysis** on model performance.
- Highlights how **outliers and multicollinearity** affect regression accuracy.

### **Multiple Linear Regression Notebook**

- Explores **feature selection** and how adding variables affects model performance.
- Emphasizes the **importance of removing correlated predictors** to prevent **multicollinearity**.
- Discusses model improvement using **interaction terms and polynomial transformation**.

---

## **Key Takeaways**

✅ **Regression is a fundamental supervised learning technique used for both continuous and categorical prediction.**  
✅ **Linear regression assumes a straight-line relationship, while non-linear regression captures complex patterns.**  
✅ **Logistic regression transforms outputs into probabilities, making it useful for binary classification.**  
✅ **Model evaluation metrics like MSE, R-squared, and Log-Loss are crucial for assessing performance.**  
✅ **Gradient Descent and its variations optimize regression models efficiently.**
