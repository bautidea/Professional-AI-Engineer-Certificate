# 📘 Module 3: Linear Regression PyTorch Way

## Section 3: Training, Validation, and Test Split

This section focuses on the critical role of dataset splitting for training machine learning models with PyTorch. It introduces best practices for managing overfitting, evaluating hyperparameters using validation data, and selecting optimal models that generalize well to unseen inputs.

## 📌 Topics and Concepts

### 🧠 Overfitting and Data Splitting

- Overfitting happens when a model learns training data too well, including its noise or outliers, leading to poor generalization.
- To avoid this, data is split into three subsets:
  - **Training set**: used to fit the model and learn parameters like weights and bias.
  - **Validation set**: used to tune hyperparameters like learning rate and batch size.
  - **Test set**: used to assess final model performance on unseen data.
- Splitting is usually random, but for demonstration purposes it may be deterministic to highlight effects like outlier influence.

### 🎯 Training vs. Hyperparameter Tuning

- Parameters (e.g., slope and bias) are learned by minimizing the loss function through gradient descent.
- Hyperparameters (e.g., learning rate, batch size) are manually chosen and significantly affect training behavior.
- Training involves fitting the model using one set of hyperparameters.
- The validation set helps compare different hyperparameter values by selecting the model that minimizes validation loss, not training loss.

### 📉 Cost Evaluation: Validation vs. Training Loss

- Validation cost is a better indicator of generalization performance than training cost.
- A model that minimizes training loss may overfit and generalize poorly.
- Example visualization:
  - A learning rate of 0.1 achieves the lowest training loss but the **highest validation loss**, leading to poor fit on validation data.
  - A learning rate of 0.001 produces higher training loss but **lowest validation loss**, and the model generalizes better to unseen data.

### 🛠️ PyTorch Implementation: Training, Validation, and Saving

- Custom dataset is built with flags to return either training or validation samples.
- Training data includes outliers to illustrate the overfitting risk.
- Training loop:
  - For each learning rate, initialize a model and optimizer.
  - Train for 10 epochs.
  - Compute and store both training and validation losses.
- Model selection:
  - Validation losses are plotted for all learning rates.
  - The model with **lowest validation loss** is selected.
  - Predictions of all trained models on validation data are visualized to confirm generalization.
- For small datasets, `dataset.x` and `dataset.y` are used directly. For larger datasets, a DataLoader should be used.

## ✅ Takeaways

- ✅ Overfitting occurs when a model fits the training data too closely and fails on unseen data.
- ✅ Splitting data into training, validation, and test sets supports robust evaluation and model tuning.
- ✅ Hyperparameters like learning rate should be selected based on **validation loss**, not training loss.
- ✅ PyTorch training includes:
  - Forward pass
  - Loss computation
  - Backpropagation
  - Parameter updates with `optimizer.step()`
- ✅ The best-performing model is the one that minimizes validation loss and aligns best with validation data.
