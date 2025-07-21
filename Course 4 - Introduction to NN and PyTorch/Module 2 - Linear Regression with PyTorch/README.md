# Module 2: Linear Regression with PyTorch

This module provides a foundational understanding of linear regression models implemented using PyTorch. It explores both the mathematical concepts and the engineering tools required to build, train, and optimize these models using PyTorch’s tensor framework, neural network layers, and autograd features.

---

## Topics and Concepts

### 1. Linear Regression Prediction

- Explains how linear regression models estimate the relationship between a predictor (x) and a target variable (y) using the equation of a line.
- Describes the role of bias (intercept) and weight (slope) as parameters to be learned.
- Introduces PyTorch’s `nn.Linear` layer for modeling this linear relationship.
- Explains how forward propagation maps input tensors to predicted outputs.
- Shows how to use PyTorch's object-oriented module system (`nn.Module`) to encapsulate and structure models using custom classes.

### 2. Linear Regression Training

- Defines the training dataset as a collection of ordered (x, y) pairs.
- Introduces the concept of noise in real-world data and how it is modeled using Gaussian distributions.
- Explains the goal of fitting a linear function that minimizes the error between predictions and actual values.
- Introduces the mean squared error (MSE) as a cost function for evaluating model performance.
- Describes how minimizing the cost function with respect to model parameters helps identify the best-fitting line.

### 3. Linear Regression Training with PyTorch (Slope Only)

- Demonstrates gradient descent in PyTorch by manually updating a single parameter: the slope.
- Uses `.backward()` and `.grad` to compute and retrieve parameter gradients.
- Highlights the importance of resetting gradients (`.grad.zero_()`) to avoid accumulation.
- Tracks how the loss decreases over multiple training iterations (epochs).
- Visualizes cost changes using Matplotlib to monitor convergence behavior.

### 4. Linear Regression Training with PyTorch (Slope and Bias)

- Expands the training process to optimize both slope and bias simultaneously.
- Visualizes the cost surface as a 3D function of slope and bias.
- Uses contour plots and slices of the cost surface to explain how gradient descent navigates parameter space.
- Discusses how gradients point in the direction of steepest descent and how they guide updates toward the minimum.
- Reinforces the concept of gradients as vectors composed of partial derivatives with respect to each parameter.

---

## Implementation Highlights

- Uses `requires_grad=True` to enable automatic differentiation in PyTorch.
- Models are defined using both `nn.Linear` layers and custom subclasses of `nn.Module`.
- Training loops include:
  - Manual gradient computation
  - Parameter updates using gradient descent
  - Resetting gradients after each step
- Visual tools like Matplotlib are used to:
  - Display the cost surface
  - Track loss over time
  - Show model predictions improving over epochs

---

## Takeaways

- Linear regression in PyTorch builds foundational skills for working with neural networks.
- PyTorch provides low-level tensor manipulation and high-level abstractions (`nn.Module`, `nn.Linear`) for flexibility in model design.
- Gradient descent is central to learning model parameters by minimizing a cost function.
- Visualizing the cost surface and training process reinforces intuitive understanding of optimization dynamics.
- By extending training to both slope and bias, the module connects basic regression to more advanced multilayer models.
