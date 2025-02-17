# 📊 Machine Learning with Python - Module 2: Regression Analysis

## 📖 Overview

This module covers **Regression Analysis**, a fundamental technique in **supervised learning** used to model relationships between **independent variables (predictors)** and a **continuous dependent variable (target)**. Regression is widely used in **forecasting, prediction, and trend analysis** across various fields such as **economics, finance, healthcare, and engineering**.

The module explores different types of regression models, including:

- **Simple Linear Regression**
- **Multiple Linear Regression**
- **Polynomial Regression**
- **Non-Linear Regression**

Additionally, two **Jupyter Notebooks** were developed as part of the hands-on practice:

- 📓 `Simple-Linear-Regression.ipynb`
- 📓 `Multiple-Linear-Regression.ipynb`

These notebooks apply regression models to real-world datasets, analyzing relationships between **engine size, fuel consumption, and CO₂ emissions**.

---

## 📌 Key Concepts in Regression

### 🔹 1️⃣ What is Regression?

Regression is a **statistical method** used to model the relationship between one or more independent variables (**predictors**) and a dependent variable (**target**). The primary goal is to find the best-fit function that **minimizes the error** in predictions.

- **Regression Equation** (General Form):  
  **ŷ = f(X) + ε**, where:

  - **ŷ** is the predicted value,
  - **X** represents the independent variables,
  - **ε** is the error term.

- **Applications of Regression:**
  - **Predicting sales based on advertising spend**
  - **Estimating house prices based on location and size**
  - **Forecasting stock prices or economic trends**
  - **Medical diagnosis based on patient data**
  - **Predicting energy consumption based on temperature changes**

---

### 🔹 2️⃣ Simple Linear Regression

A **Simple Linear Regression** model predicts a target variable using a **single independent variable**. The relationship is modeled using a **straight line**.

- **Mathematical Equation:**  
  **ŷ = θ₀ + θ₁x**

  - **θ₀**: Intercept (where the line crosses the y-axis)
  - **θ₁**: Slope (rate of change of y with respect to x)
  - **x**: Independent variable (predictor)

- **Key Features:**

  - Assumes a **linear relationship** between x and y.
  - Uses the **Ordinary Least Squares (OLS) method** to find the best-fit line by minimizing the sum of squared errors.
  - **Evaluation Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.

- **Example Application:**
  - Predicting **CO₂ emissions** based on **engine size**.
  - **Key finding:** Larger engine sizes are associated with higher CO₂ emissions.

---

### 🔹 3️⃣ Multiple Linear Regression

Multiple Linear Regression extends Simple Linear Regression by incorporating **two or more independent variables**.

- **Mathematical Equation:**  
  **ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ**

  - Each predictor (**x₁, x₂, ... xₙ**) has an associated weight (**θ**) that determines its contribution to the prediction.

- **Key Features:**

  - Useful for **modeling more complex relationships** between variables.
  - Can identify which independent variables have the **strongest impact** on the target variable.
  - Risk of **multicollinearity**, where independent variables are highly correlated, leading to unreliable coefficient estimates.

- **Example Application:**
  - Predicting **CO₂ emissions** using multiple factors:
    - **Engine size**
    - **Fuel consumption**
    - **Number of cylinders**
  - **Key finding:** CO₂ emissions are influenced by multiple factors, but some variables are strongly correlated, requiring careful feature selection.

---

### 🔹 4️⃣ Polynomial Regression

Polynomial Regression is used when a **linear model is insufficient** to capture the relationship between variables.

- **Mathematical Equation (Cubic Example):**  
  **ŷ = θ₀ + θ₁x + θ₂x² + θ₃x³**

  - Adds higher-degree polynomial terms to the model to capture curvature in the data.

- **Key Features:**

  - Allows for **better fit to complex patterns** in the data.
  - At risk of **overfitting** if too many polynomial terms are included.
  - Requires careful selection of polynomial degree.

- **Example Application:**
  - **Stock market trends** (non-linear patterns in stock prices)
  - **Modeling vehicle fuel efficiency** (efficiency often follows a curved relationship with speed)

---

### 🔹 5️⃣ Non-Linear Regression

Used when the relationship between independent and dependent variables is **not a straight line** and cannot be modeled using polynomial regression.

- **Types of Non-Linear Models:**

  - **Exponential Growth Model:**  
    **ŷ = θ₀ \* e^(θ₁x)**
    - Example: **Population growth, GDP increase over time**
  - **Logarithmic Model:**  
    **ŷ = θ₀ + θ₁ log(x)**
    - Example: **Diminishing returns in economics or productivity**
  - **Periodic Model:**  
    **ŷ = θ₀ + θ₁ sin(x)**
    - Example: **Seasonal variations in weather data**

- **Example Application:**
  - **GDP growth trends**
  - **Medical response to drug dosage** (diminishing effects over time)
  - **Weather patterns and seasonal forecasting**

---

## 📌 Key Takeaways

✅ Regression models are **essential for predicting continuous values**.  
✅ **Linear regression** works well for simple relationships but has limitations.  
✅ **Multiple regression** provides better accuracy but requires feature selection.  
✅ **Polynomial regression captures curves** but may lead to overfitting.  
✅ **Non-linear regression is necessary for exponential, logarithmic, and periodic trends**.

---

## 📂 Developed Notebooks

Two Jupyter notebooks were created for hands-on practice with regression models:

- 📓 **Simple-Linear-Regression.ipynb**: Implements a single-variable regression model to predict **CO₂ emissions**.
- 📓 **Multiple-Linear-Regression.ipynb**: Extends the analysis using multiple independent variables to improve prediction accuracy.

---
