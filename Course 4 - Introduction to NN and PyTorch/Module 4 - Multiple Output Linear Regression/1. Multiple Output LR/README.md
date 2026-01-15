# 📦 Section 4: Multiple Input / Output Linear Regression

## Multiple Output Linear Regression

### Introduction

This section focuses on **extending linear regression to handle multiple outputs simultaneously** using PyTorch. Unlike single-output models, where the prediction is a scalar, multiple-output models return a vector of predictions for each input sample. The section covers the mathematical formulation, PyTorch implementation using both `nn.Linear` and custom modules, and the complete training workflow for multi-output models. The concepts here are foundational for more advanced neural network architectures where multiple targets or output dimensions are common.

### Topics and Concepts

#### **Multiple Output Linear Functions**

- **Concept**: A single model predicts multiple dependent variables at once by applying several linear transformations in parallel.
- **Mathematical Structure**:
  - **Weight matrix (W)**: Shape `D × M`, where:
    - `D` = number of input features (columns in X).
    - `M` = number of outputs (columns in Y).
    - Each column of **W** corresponds to the weights for one output.
  - **Bias vector (b)**: Shape `1 × M`, with one bias term per output.
- **Prediction Process**:
  1. Take the dot product between the input vector **x** and each column of **W**.
  2. Add the corresponding bias from **b**.
  3. The result is a **1 × M** tensor of predictions.
- **Graph Representation**:
  - Nodes represent input features.
  - Edges represent learned weights and biases.
  - Multiple output nodes allow visualizing separate prediction paths from the same inputs.
- **Batch Predictions**:
  - For **N** samples in matrix **X** (shape `N × D`), the prediction matrix **Ŷ** has shape `N × M`.
  - Biases are added across all samples using **broadcasting** in PyTorch.

#### **Custom Modules for Multi-Output Models**

- Built using `nn.Module` with an internal `nn.Linear(in_features, out_features)` layer.
- **Key Parameters**:
  - `in_features`: Number of input features (size of each sample).
  - `out_features`: Number of predicted values per sample.
- **Prediction Behavior**:
  - Single sample: Input shape `(1, D)` → Output shape `(1, M)`.
  - Multiple samples: Input shape `(N, D)` → Output shape `(N, M)`.
- Benefits:
  - Flexibility for varying input-output dimensions.
  - Clean integration into larger neural network architectures.
  - Simplified parameter handling via `model.parameters()` and `state_dict()`.

#### **Training Multiple Output Models in PyTorch**

- **Targets (y)**: Each target is a vector containing M values.
- **Predictions (ŷ)**: Model outputs vectors of the same size for each sample.
- **Cost Function**:
  - Mean Squared Error (MSE) applied across all outputs per sample, summed over the batch.
- **Parameters**:
  - **W** is a matrix (one column per output).
  - **b** is a bias vector with one value per output.
- **Dataset**:
  - Generates multiple target values for each input.
  - Works with both custom dataset classes and PyTorch’s built-in structures.
- **Training Loop**:
  1. Initialize the model with correct input and output dimensions.
  2. Define the loss function (e.g., `nn.MSELoss`).
  3. Create an optimizer (e.g., `torch.optim.SGD`) and set the learning rate.
  4. Loop through epochs:
     - Retrieve batches from the DataLoader.
     - Forward pass: compute predictions for the batch.
     - Calculate loss.
     - Reset gradients with `optimizer.zero_grad()`.
     - Backward pass with `.backward()` to compute gradients.
     - Update parameters with `optimizer.step()`.

### Takeaways

- Multiple-output regression models share the same underlying process as single-output models but operate with **matrix weights** and **vector biases**.
- Changing the `out_features` parameter in PyTorch’s `nn.Linear` or a custom module makes it easy to scale from one to many outputs.
- Matrix multiplication and broadcasting ensure efficient computation across all samples and outputs.
- This structure is essential for multi-task learning, multi-output prediction problems, and more complex neural network designs.
