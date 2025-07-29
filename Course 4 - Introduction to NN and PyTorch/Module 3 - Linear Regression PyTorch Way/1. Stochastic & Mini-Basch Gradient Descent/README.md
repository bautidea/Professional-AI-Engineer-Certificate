# 🧠 Module 3: Linear Regression PyTorch Way

## Section 1: Stochastic Gradient Descent and Data Loader

This section introduces foundational techniques for training neural networks using PyTorch, focusing on **Stochastic Gradient Descent (SGD)** and **Mini-Batch Gradient Descent**, as well as the use of the **DataLoader** for efficient data handling.

## 🔹 Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent is an optimization technique that updates model parameters one data sample at a time. This allows faster updates and lower memory consumption but introduces fluctuations due to sample-level noise.

- Parameter updates are computed after every single data point.
- Model predictions (lines) fluctuate between iterations, especially when data points are outliers.
- The cost function is approximated by computing the loss for each sample.
- Progress is tracked using loss values stored per iteration or per epoch.

PyTorch Implementation Highlights:

- Gradients are computed using `.backward()` and parameters updated manually.
- Loss values are accumulated and tracked for convergence analysis.
- Training loops are written explicitly using nested iterations over samples and epochs.

## 🔹 DataLoader for SGD

PyTorch’s `DataLoader` simplifies iteration by batching and shuffling data.

- Requires a custom `Dataset` class implementing `__getitem__` and `__len__`.
- Allows seamless access to data via indexing and slicing.
- Supports stochastic gradient descent when `batch_size=1`.

Training with DataLoader:

- Each batch (in this case, a single sample) is fed through the model.
- Loss is computed and backpropagated.
- Parameter updates are applied using gradients from each sample.

Using DataLoader provides:

- Scalable sample management
- Easier integration into PyTorch pipelines
- Configurable batch processing and shuffling

## 🔹 Mini-Batch Gradient Descent

Mini-batch gradient descent balances the noise of SGD and the stability of batch gradient descent.

- The dataset is split into small subsets (batches).
- Each batch computes a "mini" cost function for parameter updates.
- More stable than SGD, and more efficient than full-batch descent.

Key Concepts:

- The number of iterations per epoch = total samples / batch size
- Smaller batches = more frequent updates, faster convergence but noisier
- Larger batches = smoother convergence but require more computation

PyTorch Implementation:

- Same as SGD but with `batch_size > 1` in the `DataLoader`
- Batches are passed through the model
- Loss is computed for each batch and parameters are updated accordingly

Visualization:

- Cost vs. iteration plots help evaluate convergence behavior
- Different batch sizes show varying convergence rates and stability

## ✅ Takeaways

- **SGD** enables fast, sample-wise parameter updates but introduces noise.
- **Mini-batch GD** processes small groups of samples, improving efficiency and convergence stability.
- **DataLoader** is essential for scalable and maintainable data pipelines in PyTorch.
- **Loss tracking** per iteration or epoch is critical for understanding model progress.
- **Batch size** directly affects convergence rate and training dynamics.
