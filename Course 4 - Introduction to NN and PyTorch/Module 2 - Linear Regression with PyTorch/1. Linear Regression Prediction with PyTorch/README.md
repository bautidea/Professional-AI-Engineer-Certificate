# 📘 Linear Regression Prediction (Module 2 – Section 1)

This section introduces the fundamentals of linear regression in one dimension and demonstrates how linear models are implemented and used in PyTorch for prediction. It highlights both the manual approach using tensors and the use of PyTorch's built-in `nn.Linear` class for scalable and structured model development.

---

## 🔍 Core Concepts

### Linear Regression Fundamentals

Linear regression models the relationship between a single feature (`x`) and a target value (`y`) using a simple equation of a line:

ŷ = wx + b

- `w`: weight (slope) of the line
- `b`: bias (intercept)
- `ŷ`: predicted output

The training goal is to determine optimal values for `w` and `b` that best approximate the data relationship.

---

### Manual Prediction Using Tensors

PyTorch enables direct implementation of the linear equation using tensors:

- Parameters (slope and bias) are created as tensors with `requires_grad=True` to enable optimization.
- A custom forward function is defined to perform the prediction.
- This approach works for both individual inputs and batches of multiple samples.

---

### Built-in Models with `nn.Linear`

PyTorch's `nn.Linear` module simplifies model construction:

- Automatically initializes weight and bias.
- Accepts `in_features` and `out_features` as configuration.
- Treats each row of input as an individual sample for prediction.
- Internally manages the forward computation—no need to explicitly call `forward()`.

Model weights and biases can be inspected using:

- `.parameters()` – returns all trainable tensors.
- `.state_dict()` – returns a dictionary with named parameters (`linear.weight`, `linear.bias`).

---

### Creating Custom Modules

A custom linear model can be built by subclassing `nn.Module`:

- The constructor initializes a linear layer using `nn.Linear`.
- A `forward()` method defines how inputs are processed.
- Once constructed, the module behaves like a callable function for prediction.
- Supports batch input and internal parameter management via `.parameters()` and `.state_dict()`.

---

## ✅ Key Takeaways

- Linear regression defines a simple mapping between inputs and outputs using a line.
- PyTorch supports manual and modular approaches to implement predictive models.
- Built-in layers like `nn.Linear` streamline development and enable model scaling.
- Custom modules allow structured and reusable code using PyTorch’s object-oriented features.
- Tools like `.state_dict()` and `.parameters()` provide control and inspection of model internals.

This foundation prepares for training and extending linear models to handle more complex prediction tasks.
