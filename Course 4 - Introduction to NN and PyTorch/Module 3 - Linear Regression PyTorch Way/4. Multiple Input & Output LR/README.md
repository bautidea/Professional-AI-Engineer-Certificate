# Multiple Input / Output Linear Regression — PyTorch Implementation

## Introduction

This module covers the implementation of **multiple linear regression** and **multiple-output linear regression** using PyTorch.  
It focuses on understanding shape consistency, vectorized operations, parameter initialization, and efficient training using `nn.Linear`, custom modules, and PyTorch's training workflow.  
Both **prediction** and **training** processes are explained for single-output and multi-output cases, including how to adapt datasets, models, and cost functions to different dimensions.

## Topics and Concepts

### **Multiple Linear Regression Prediction**

- **Concept**: Extends single-variable regression to handle multiple input features.
- **Mathematics**:
  - Input `x`: 1×D tensor (feature vector).
  - Weights `w`: D×1 tensor (weight vector).
  - Bias `b`: scalar added to each prediction.
  - Output `ŷ`: computed as `x·w + b`.
- **Batch Predictions**:
  - An N×D input matrix produces an N×1 output vector.
  - Each row of `X` is multiplied by `w` and `b` is added.
- **Implementation with `nn.Linear`**:
  - `in_features`: number of input columns (features).
  - `out_features`: number of outputs (1 for regression).
  - Parameters can be accessed with `.parameters()` or `.state_dict()`.
- **Custom Modules**:
  - Created by subclassing `nn.Module`.
  - Contain an internal `nn.Linear` layer.
  - Implement the `forward()` method for predictions.
  - Behave identically to `nn.Linear` but allow more flexibility.

### **Multiple Linear Regression Training**

- **Cost Function**: Mean Squared Error (MSE) between predicted `ŷ` and target `y`.
- **Gradient Descent**: Updates weights and bias based on partial derivatives.
- **PyTorch Workflow**:
  1. Create dataset and DataLoader for batching.
  2. Define the model (using `nn.Linear` or custom class).
  3. Define the loss function (criterion).
  4. Initialize optimizer (e.g., SGD with learning rate).
  5. Training loop:
     - Forward pass → compute predictions.
     - Compute loss.
     - Zero gradients (`optimizer.zero_grad()`).
     - Backward pass (`loss.backward()`).
     - Update parameters (`optimizer.step()`).
- **Outcome**: After enough epochs, the learned plane (in 2D input space) fits the training data more accurately.

### **Multiple Output Linear Regression**

- **Concept**: Produces multiple outputs from the same input features.
- **Mathematics**:
  - Weights `W`: D×M matrix (D features, M outputs).
  - Bias `b`: 1×M vector.
  - Output `ŷ`: 1×M vector per sample.
- **Prediction Process**:
  - Dot product of `x` with each column of `W`.
  - Add corresponding bias to each result.
- **Batch Predictions**:
  - Input: N×D matrix.
  - Output: N×M matrix.
  - Rows → input samples; Columns → output dimensions.
- **Custom Modules**:
  - Same structure as single-output version, but `out_features > 1`.
  - Broadcasting used to add bias to all samples efficiently.

### **Multiple Output Linear Regression Training**

- **Cost Function**: Sum of squared differences between predicted and target vectors, across all outputs.
- **PyTorch Workflow**:
  1. Dataset generates multi-output targets.
  2. DataLoader handles batching.
  3. Model initialized with `in_features` and `out_features`.
  4. Loss function compares predictions and targets over all outputs.
  5. Training loop is identical to single-output case, except tensor shapes differ.
- **Parameter Updates**:
  - Weight matrix `W` and bias vector `b` updated simultaneously using vectorized operations.
  - Gradients computed for all outputs in a single backward pass.

## Key Takeaways

- Multiple linear regression generalizes to **D-dimensional inputs** using dot products and bias addition.
- Multiple-output regression uses a **matrix of weights** and a **vector of biases** to produce several predictions at once.
- `nn.Linear` efficiently handles both cases, with easy parameter inspection via `.parameters()` or `.state_dict()`.
- Custom modules offer flexibility for future extension into complex neural networks.
- Training process in PyTorch is consistent across single and multi-output models — changes mainly involve tensor shapes and output dimensions.
- Vectorized operations ensure efficient computation for both prediction and training.
