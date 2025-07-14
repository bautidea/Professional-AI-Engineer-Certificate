# 📘 Module 2 – Linear Regression with PyTorch

## Section 2: Linear Regression Training

This section focuses on the process of training linear regression models using labeled datasets. It explores how model parameters (slope and bias) are learned from data through optimization techniques and introduces the foundations of supervised learning, noise modeling, and cost minimization.

---

## 🔹 Supervised Learning and Data Modeling

Linear regression is a foundational supervised learning technique where the model is trained on input-output pairs to learn a predictive function.

- The **input variable (x)** is the independent variable or feature.
- The **output variable (y)** is the dependent variable or label.

The training dataset consists of **N pairs**:  
\[
(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)
\]

The objective is to discover a linear mapping:
\[
ŷ = wx + b
\]
Where:

- `w` is the **weight** (slope)
- `b` is the **bias**
- `ŷ` is the predicted value of y

This learned function allows prediction of `y` given new, unseen values of `x`.

---

## 🔹 Real-World Examples of Simple Linear Regression

- **Housing prices**: Predict `price` (y) from `square footage` (x)
- **Stock forecasting**: Predict `stock price` from `interest rate`
- **Vehicle efficiency**: Predict `fuel economy` from `horsepower`

These examples involve continuous target variables and one feature, making them ideal for simple linear regression.

---

## 🔹 Noise and the Gaussian Assumption

Even when a linear relationship exists, real-world data contains **error** or **noise**. This section introduces the assumption that:

- Noise is **random**, **normally distributed**, and **centered at zero**.
- The **standard deviation** of the distribution determines how much data deviates from the linear function.

The visual interpretation:

- Most noise values are small and cluster around zero.
- Large deviations are rare.
- Greater variance results in wider dispersion of data points.

This noise assumption is essential to understanding why linear regression does not perfectly fit all data but aims to approximate the underlying pattern.

---

## 🔹 Training Objective and Error Minimization

The model must find the best-fitting line through the training data. This is done by minimizing the total prediction error using a **cost function**.

### 🔸 Mean Squared Error (MSE)

The cost function is defined as:
\[
L(w, b) = \frac{1}{N} \sum\_{i=1}^N (\hat{y}\_i - y_i)^2
\]

Where:

- \( \hat{y}\_i \) is the model's prediction for sample i
- \( y_i \) is the actual value
- The expression is averaged across all training samples

This function penalizes large prediction errors more heavily and provides a scalar measure of model performance.

---

## 🔹 Parameter Optimization

The slope (`w`) and bias (`b`) are **trainable parameters**. During training:

- The model makes predictions using current values of `w` and `b`
- The cost is calculated based on how far off the predictions are
- An optimization algorithm (gradient descent, discussed next) adjusts the parameters to reduce cost

This iterative adjustment continues until the model achieves the best possible fit to the data or satisfies stopping criteria.

---

## ✅ Key Takeaways

- Training linear regression involves learning the optimal slope and bias that minimize prediction errors across a dataset.
- Supervised learning uses known input-output pairs to fit the model.
- Real-world data includes Gaussian noise, and the model accounts for this during training.
- The cost function used is the Mean Squared Error (MSE), which quantifies the average prediction error.
- Training prepares the model to generalize well on new data by fitting the underlying trend in the training set.

This section provides the foundation for understanding how model parameters are updated, setting the stage for optimization with gradient descent.
