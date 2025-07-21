# 📘 Module 2 – Linear Regression with PyTorch

## Section 3: PyTorch Slope

This section presents a manual approach to training a linear regression model using PyTorch, focusing on learning the slope parameter through raw gradient descent. The objective is to develop a deeper conceptual understanding of how model parameters are updated iteratively using gradients, without relying on high-level PyTorch abstractions.

### 🔹 Manual Gradient Descent with Tensors

The slope parameter is defined as a PyTorch tensor with `requires_grad=True`, enabling automatic differentiation during the optimization process. A dataset is created by generating synthetic `X` values and computing their corresponding `Y` targets using a linear function. Random noise is added to simulate real-world data variability.

- The `view()` method is used to reshape data into column vectors.
- The line and noisy samples are visualized using matplotlib.
- The forward function performs a simple linear transformation representing the model.

### 🔹 Loss Computation and Parameter Updates

Training involves calculating the Mean Squared Error (MSE) between predicted and actual values. Though it represents overall cost, the term "loss" is used to stay consistent with PyTorch's API conventions.

The optimization steps include:

- Generating predictions using the forward function.
- Computing the loss and invoking `.backward()` to calculate gradients.
- Accessing gradients with `.grad` and updating parameters with `.data`.
- Resetting gradients after each iteration using `.zero_()` to avoid accumulation.

### 🔹 Epochs, Iterative Updates, and Loss Convergence

The model is trained over multiple epochs. Each epoch consists of a full pass over the data, performing the update step and reducing the average loss:

- The red dot in the cost function plot shows the current parameter value.
- The predicted line (in blue) progressively aligns with the noisy data (in red).
- Early iterations have steep gradients, leading to larger updates.
- Later iterations converge more slowly as the model approaches an optimal slope.

The process reflects the natural diminishing updates of gradient descent as the loss approaches its minimum.

### 🔹 Monitoring Loss Progression

To analyze convergence:

- The average loss is recorded at each epoch using `.item()` to extract scalar values.
- The loss values are visualized over time to track learning progress.
- The plot reveals a consistent reduction in cost and smoother convergence patterns.

This approach provides transparency into the optimization process and allows visual inspection of model improvement across iterations.

### ✅ Key Takeaways

- Manual gradient descent in PyTorch reinforces core concepts of training and optimization.
- Automatic differentiation via `requires_grad=True` simplifies gradient computation.
- Parameter updates rely on `.grad` access and direct tensor manipulation with `.data`.
- Iterative training over epochs reveals how gradient magnitude affects learning speed.
- Tracking loss per iteration supports interpretability and debugging of convergence.
- This foundational method builds practical understanding for more advanced PyTorch workflows.
