# Module 4: Multiple Input / Output Linear Regression with PyTorch

This module focuses on extending linear regression models to handle multiple inputs and multiple outputs using PyTorch. It explores how matrix-based formulations, tensor shapes, and vectorized operations allow efficient prediction and training when working with higher-dimensional data.

---

## Topics and Concepts

### 1. Multiple Output Linear Regression

This section introduces linear regression models that produce multiple outputs simultaneously.

- Explains how multiple linear functions can be computed in parallel using a single weight matrix and bias vector.
- Describes the role of the weight matrix **W (D × M)**, where each column corresponds to a different output.
- Shows how predictions are computed using matrix multiplication and bias addition.
- Explains how dot products and broadcasting enable efficient multi-output predictions.
- Uses directed graph representations to illustrate how inputs flow through weights and biases to produce multiple outputs.

### 2. Custom Modules for Multi-Output Models

This section focuses on building extensible models using PyTorch’s module system.

- Uses `nn.Linear(input_dim, output_dim)` to support multiple outputs directly.
- Shows how custom classes inheriting from `nn.Module` behave like built-in layers while allowing full customization.
- Explains prediction behavior for:
  - Single input samples (1 × D → 1 × M)
  - Batch inputs (N × D → N × M)
- Highlights how matrix operations scale naturally across samples and outputs.

### 3. Training Multiple Output Linear Regression Models

This section explains how training generalizes when models produce vector-valued outputs.

- Describes how the cost function aggregates squared errors across all output dimensions.
- Explains how weights become matrices and biases become vectors in multi-output settings.
- Shows that gradient descent remains conceptually identical, with updates applied in a vectorized form.
- Covers the full PyTorch training loop:
  - Forward pass
  - Loss computation
  - Gradient reset
  - Backward pass
  - Parameter updates
- Highlights that all outputs are optimized simultaneously during each update step.

### 4. Multiple Linear Regression with Multiple Inputs

This section shifts focus to handling higher-dimensional input features.

- Explains how predictions are computed using dot products between feature vectors and weight vectors.
- Emphasizes the importance of shape consistency between inputs and parameters.
- Describes how predictions scale from single samples to batch inputs.
- Explains how PyTorch applies the same bias across all samples during batch prediction.

### 5. Linear Regression Using `nn.Linear`

This section covers PyTorch’s built-in linear layer in detail.

- Explains how `nn.Linear` initializes weights and bias automatically.
- Describes how parameters are stored and accessed using `.parameters()` and `.state_dict()`.
- Shows how the same layer supports:
  - Single-sample predictions
  - Batch predictions
- Highlights how `nn.Linear` abstracts low-level tensor operations while remaining fully transparent.

### 6. Training Multiple Linear Regression Models in PyTorch

This section presents the complete training pipeline for multi-dimensional regression.

- Explains how the squared error loss generalizes to higher-dimensional inputs.
- Describes gradient computation for both weights and bias in vector form.
- Covers the use of:
  - Custom model classes
  - Dataset abstractions
  - DataLoader for mini-batch training
- Explains how PyTorch’s autograd system computes gradients automatically.
- Shows how vectorized updates improve efficiency and scalability during training.

---

## Takeaways

- Multi-output linear regression produces multiple predictions using a single matrix of weights and a bias vector.
- PyTorch’s `nn.Linear` and custom modules naturally support both multiple inputs and multiple outputs.
- Matrix operations and broadcasting are essential for efficient prediction and training.
- Training loops remain structurally identical to single-output regression, with changes limited to tensor shapes.
- Custom modules built with `nn.Module` provide flexibility while maintaining full compatibility with PyTorch’s API.
