# 📘 Module 5: Linear Regression PyTorch Way

## ℹ️ Overview

This module focuses on **logistic regression** as one of the most important linear models for classification. While linear regression predicts continuous values, logistic regression is designed to predict **class membership**, making it fundamental for supervised learning tasks where outputs are discrete. The module explains the foundations of linear classifiers, introduces the **logistic function**, and develops a full understanding of how logistic regression is connected to probability, the **Bernoulli distribution**, and **maximum likelihood estimation (MLE)**. Finally, the module introduces **cross-entropy loss** as the cost function that enables effective training of classification models in PyTorch.

---

## 🧩 Topics and Concepts

### 📝 Linear Classifiers and Logistic Regression

Linear classifiers define a decision boundary—typically a line in one dimension, a plane in two dimensions, or a hyperplane in higher dimensions—that separates samples of different classes.

- A sample is represented by a feature vector `x`, and classification is achieved by applying a weighted sum with bias, expressed as `z = w · x + b`.
- If `z > 0`, the model assigns one class; if `z < 0`, it assigns another.

However, raw linear outputs are real numbers, not probabilities. Logistic regression solves this by applying the **sigmoid (logistic) function**, which maps values of `z` smoothly into the range [0, 1]. This allows predictions to be interpreted as probabilities and enables robust classification even for data that is not perfectly separable.

### 📝 Logistic Regression in PyTorch

In PyTorch, logistic regression can be implemented in different ways:

- **Using `nn.Sigmoid` or `torch.sigmoid()`**: Both approaches apply the logistic function to map linear outputs into probabilities.
- **Using `nn.Sequential`**: A compact way to build models by stacking a `Linear` layer followed by a `Sigmoid` activation. Data flows automatically through this sequence: input → linear transformation → sigmoid → probability.
- **Using custom `nn.Module` classes**: This approach provides more control and flexibility, defining the linear transformation and sigmoid activation explicitly in a `forward` method.

Models can handle both single-sample and multi-sample inputs, and they scale naturally from 1D to multi-dimensional feature spaces. Logistic regression with PyTorch thus combines conceptual clarity with practical flexibility.

### 📝 Bernoulli Distribution and Maximum Likelihood Estimation

Logistic regression has a strong **probabilistic foundation**. Binary outcomes, such as success/failure or class 0/class 1, are modeled using the **Bernoulli distribution**, parameterized by a single probability θ.

- The probability of outcome `y = 1` is θ, while the probability of `y = 0` is `1 − θ`.
- For a dataset of independent samples, the **likelihood function** is the product of all probabilities under a given θ.

Training a logistic regression model is equivalent to finding the parameters (weights and bias) that **maximize the likelihood** of the observed data. In practice, this is achieved by maximizing the **log-likelihood**, which simplifies calculations and aligns naturally with gradient-based optimization.

### 📝 Cross-Entropy Loss

While mean squared error works well for regression tasks, it fails in classification because it creates flat cost surfaces that block parameter updates. Instead, classification uses **cross-entropy loss**, derived directly from the negative log-likelihood of the Bernoulli distribution.

- Cross-entropy measures the difference between predicted probabilities and true labels.
- It penalizes confident but incorrect predictions more heavily than uncertain ones, guiding the model to align its probabilities with the true data distribution.
- Unlike MSE, cross-entropy provides smooth gradients, ensuring stable convergence during optimization.

In PyTorch, cross-entropy loss for logistic regression can be implemented with `nn.BCELoss`, and training proceeds with the usual gradient descent steps: forward pass, loss calculation, backpropagation, and parameter updates.

---

## ✅ Takeaways

- Logistic regression extends linear classifiers by mapping outputs into probabilities using the sigmoid function.
- The Bernoulli distribution provides the statistical foundation for binary classification, with maximum likelihood estimation guiding parameter learning.
- Cross-entropy loss, derived from the log-likelihood, is the key to effective training, avoiding the pitfalls of mean squared error in classification.
- PyTorch provides multiple tools (`nn.Sequential`, `nn.Module`, `nn.Sigmoid`, `nn.BCELoss`) that make implementing logistic regression models straightforward and scalable.
- This module establishes a complete path from **theory (probability and MLE)** to **implementation (logistic regression training in PyTorch)**, forming a cornerstone for building more advanced classification models.
