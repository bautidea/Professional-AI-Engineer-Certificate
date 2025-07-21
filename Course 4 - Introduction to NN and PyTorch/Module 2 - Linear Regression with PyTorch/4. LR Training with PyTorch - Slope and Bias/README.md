# 📘 Linear Regression Training with PyTorch – Slope and Bias

This section focuses on manually optimizing both the slope and bias parameters in a linear regression model using PyTorch. It demonstrates how to visualize and minimize the cost function and provides insight into gradient-based parameter updates when working with multivariate functions.

## 🔍 Topics and Concepts

### 📐 Cost Surface in Linear Regression

The cost surface represents the total or average prediction error across all training samples. It is defined as a function of both slope and bias, and its shape can be visualized as a 3D surface:

- One axis corresponds to the slope (weight).
- One axis corresponds to the bias (intercept).
- The vertical axis represents the cost (error).

This surface is used to understand how prediction error changes in response to different combinations of slope and bias values. The minimum point on this surface corresponds to the optimal parameter values.

### 🔗 Contour Plot Representation

A contour plot is used to view the cost surface from above. Each line on the plot connects parameter combinations that produce the same error. These lines provide a geometric interpretation of how steep or shallow different regions of the cost surface are:

- Closer lines represent steep regions (rapid error change).
- Wider lines represent flatter regions (slow error change).

Contour slicing along one axis allows isolation and inspection of how the cost changes with one parameter while keeping the other fixed.

### 🔁 Manual Gradient Descent with Slope and Bias

The training process uses manual gradient descent to update both slope and bias:

- The forward function defines the linear model that maps input to output using the current slope and bias.
- The cost function measures prediction error and is treated as a function of both parameters.
- PyTorch's automatic differentiation tracks how the loss changes with respect to both parameters.
- Partial derivatives of the cost function with respect to slope and bias are calculated.
- Parameter updates are applied iteratively, subtracting a scaled version of the gradient from the current values.

Each update moves the parameters toward the region of lowest cost, which results in a better-fitting regression line.

### 🧭 Gradient Vector and Optimization Direction

The gradient is a vector containing the partial derivatives of the cost function with respect to each parameter:

- The gradient points in the direction of steepest increase.
- The negative gradient is used to update the parameters and minimize the cost.

This approach ensures that each iteration moves toward reducing the overall prediction error. The update direction is always aligned with the steepest descent path in the multivariate parameter space.

### 📉 Training Behavior Over Iterations

The cost surface is sampled at each iteration, showing the progress of parameter updates:

- The model starts with arbitrary initial values for slope and bias.
- After each iteration (epoch), the cost decreases, and the regression line aligns more closely with the data.
- This behavior is observed both in parameter space (as the red “X” marks converge) and in data space (as the predicted line fits the points better).

After several iterations, the model produces a line that minimizes the overall error and effectively captures the linear relationship in the data.

## ✅ Takeaways

- The cost surface is a two-dimensional function used to visualize prediction error in terms of both slope and bias.
- Contour plots provide a geometric interpretation of the cost landscape and guide parameter updates.
- Gradient descent is used to minimize error by iteratively updating slope and bias in the direction of the negative gradient.
- PyTorch computes partial derivatives automatically, supporting manual optimization through low-level tensor operations.
- Parameter updates over multiple epochs lead to a well-fitted model with minimized cost and improved predictive accuracy.
