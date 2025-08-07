# 📦 Module 4: Multiple Input / Output Linear Regression

## Section 1: Multiple Linear Regression Prediction

This section explores how to build and train multiple linear regression models using PyTorch. It introduces multi-dimensional input handling, efficient parameter usage, and how to define models both through built-in tools (`nn.Linear`) and custom class modules derived from `nn.Module`.

## 📚 Topics and Concepts

### 🔹 Multiple Linear Regression in Multiple Dimensions

- Multiple input features (`x₁, x₂, ..., x_D`) are linearly combined to predict a scalar target (`ŷ`).
- The linear prediction is computed using:

  ŷ = x · w + b

  where `x` is a 1×D tensor (input), `w` is a D×1 tensor (weights), and `b` is a scalar (bias).

- For batch predictions:
- `X` becomes an N×D matrix.
- Each row of `X` is processed independently.
- Output is an N×1 tensor (one prediction per row).
- Dot product understanding is reinforced through visual and color-coded representations.

### 🔹 Using `nn.Linear` for Multi-Dimensional Regression

- `nn.Linear` handles linear transformations:
- `in_features`: number of input features.
- `out_features`: typically 1 for regression tasks.
- Internally:
- Parameters (weights and bias) are initialized randomly.
- They can be accessed using `.parameters()` or `.state_dict()`.
- Single-sample and batch predictions are supported:
- 1×D input returns a 1×1 output.
- N×D input returns N×1 predictions.

### 🔹 Defining Custom Modules with `nn.Module`

- Custom regression models are defined by subclassing `nn.Module`.
- Structure:
- `__init__()` defines internal layers (e.g., `nn.Linear`).
- `forward()` defines how input is transformed to output.
- Advantages:
- Mirrors behavior of `nn.Linear`.
- Provides flexibility to expand toward complex neural network architectures.
- Model calls use standard Python call syntax (e.g., `model(x)`), which internally triggers `forward()`.

### 🔹 Multiple Linear Regression Training in PyTorch

- Training procedure generalizes gradient descent to multi-feature data:
- Loss function is squared error across samples and features.
- Weight updates are performed using vectorized operations.
- PyTorch steps:

1. Create a dataset using a custom `Dataset` class.
2. Define a model using a custom class (subclass of `nn.Module`) or `nn.Linear`.
3. Instantiate a loss function (`MSELoss`) and optimizer (`SGD`).
4. Loop over epochs and batches:
   - Forward pass: compute predictions.
   - Compute loss and call `.backward()` to compute gradients.
   - Use `optimizer.step()` to update weights.
   - Zero gradients with `optimizer.zero_grad()`.

## ✅ Takeaways

- Multiple Linear Regression extends linear modeling to D-dimensional features via dot product and scalar bias.
- `nn.Linear` automates parameter handling and supports flexible input shapes.
- Custom modules provide full extensibility for future neural network designs.
- Batched data enables efficient training across samples, and PyTorch’s autograd system simplifies gradient-based updates.
- Vectorized gradient descent in PyTorch improves scalability and model accuracy over multiple epochs.
