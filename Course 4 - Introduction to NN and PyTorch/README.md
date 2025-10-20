# 🧠 Introduction to Neural Networks with PyTorch

This repository documents my progress through the **"Introduction to Neural Networks with PyTorch"** course, part of the **IBM AI Engineering Professional Certificate**.

PyTorch is one of the most in-demand technologies in AI development. It is widely used to design, train, and optimize neural networks in domains such as image recognition, natural language processing, and predictive analytics. This course provides a practical introduction to core neural network concepts using PyTorch, reinforced by hands-on labs and a final project.

This course introduces the foundations of building and training neural networks using **PyTorch**. It begins with essential concepts such as **tensors**, **automatic differentiation**, and **datasets**, and progressively develops into more advanced topics such as **linear regression**, **gradient descent optimization**, and **classification with logistic regression**.

The progression ensures a step-by-step understanding of how PyTorch manages data, gradients, optimization, and model evaluation, preparing learners for deeper explorations into neural networks and modern deep learning workflows.

---

## 🧩 Course Modules

### 🛠 Module 1: Tensor and Dataset

This module establishes the groundwork for PyTorch by introducing tensors, differentiation, and dataset handling.

- **Tensors as Core Structures**  
  Tensors represent all forms of data in PyTorch: feature vectors, matrices, images, or parameters. Operations include indexing, slicing, reshaping, broadcasting, and matrix multiplication. They can be seamlessly converted to and from NumPy arrays, enabling compatibility with the Python data ecosystem.

- **Automatic Differentiation**  
  PyTorch dynamically constructs computation graphs, enabling automatic differentiation. Attributes such as `grad`, `grad_fn`, and `requires_grad` allow the framework to compute derivatives and backpropagate through networks. This mechanism is the foundation of optimization in neural networks.

- **Datasets and Transformations**  
  Custom datasets are built by subclassing `torch.utils.data.Dataset`. Each dataset implements `__getitem__` for indexed access and `__len__` for sample count. Transformations can be applied using callable classes or composed pipelines (`transforms.Compose`) to normalize, augment, or modify samples before training.

- **Image Datasets and TorchVision**  
  TorchVision extends PyTorch for image tasks, providing prebuilt datasets like MNIST and Fashion-MNIST, as well as transformations (cropping, scaling, tensor conversion). Images are loaded dynamically, paired with labels, and transformed into tensors ready for model input.

**Key Takeaway:** Tensors are the backbone of PyTorch, automatic differentiation enables gradient-based learning, and datasets (structured or image-based) provide the standardized input pipelines for training neural networks.

### 📊 Module 2 – Linear Regression with PyTorch

This module introduces **linear regression** in PyTorch, moving from mathematical formulation to model implementation and training.

- **Linear Regression Fundamentals**  
  Linear regression models the relationship between input features and a target variable using weights and bias. Predictions are estimates (`ŷ`) produced through the equation of a line (1D) or hyperplane (multi-D). Training involves finding the parameters that minimize the error between predictions and actual values.

- **Training with Gradient Descent**  
  The training process is framed as minimizing a **cost function** (Mean Squared Error). Gradient descent iteratively updates parameters in the direction of decreasing error. The course demonstrates how both slope and bias contribute to shaping the prediction line.

- **PyTorch Implementation**  
  Using `nn.Linear`, models are created with defined input and output sizes. PyTorch manages forward passes, gradient tracking, and parameter updates. Custom modules subclassing `nn.Module` allow greater flexibility, enabling the combination of multiple components into more complex architectures.

- **Optimization Concepts**  
  Loss functions quantify prediction error, and optimizers like `torch.optim.SGD` update parameters efficiently. The course emphasizes how cost surfaces and gradients interact, visualizing parameter updates in relation to convergence.

**Key Takeaway:** Linear regression provides the foundation for supervised learning, introducing cost functions, gradient descent, and parameter optimization in PyTorch.

### ⚡ Module 3 – Linear Regression PyTorch Way

This module extends linear regression into more **efficient PyTorch workflows**, emphasizing optimization strategies, data handling, and scaling to more complex tasks.

- **Stochastic and Mini-Batch Gradient Descent**  
  Instead of processing the entire dataset at once (batch gradient descent), stochastic gradient descent (SGD) updates parameters using one sample at a time. Mini-batch gradient descent balances efficiency and stability by processing small subsets per iteration. PyTorch’s **DataLoader** automates batching and shuffling, critical for large datasets.

- **Optimization in PyTorch**  
  PyTorch optimizers (`torch.optim.SGD`, etc.) abstract gradient descent, managing parameter updates while keeping track of state. The training loop integrates predictions, loss calculation, backpropagation, and parameter updates in a consistent and modular manner.

- **Training, Validation, and Test Splits**  
  Proper data splitting ensures that models generalize to unseen data. Training sets adjust model parameters, validation sets tune hyperparameters, and test sets evaluate final performance. The module demonstrates how overfitting and underfitting manifest and how validation guides the selection of hyperparameters like learning rate and batch size.

- **Multiple Input and Output Regression**  
  Linear regression is extended to handle multiple predictors and multiple outputs simultaneously. PyTorch’s `nn.Linear` supports arbitrary input and output dimensions, while custom modules allow explicit definition of forward passes. Training procedures remain consistent, but tensor shapes scale with dimensionality.

- **Logistic Regression for Classification**  
  The transition from regression to classification introduces logistic regression, where outputs represent class probabilities via the **sigmoid function**. Theoretical foundations include the **Bernoulli distribution**, **maximum likelihood estimation (MLE)**, and **cross-entropy loss**. Logistic regression is implemented in PyTorch using both `nn.Sequential` and custom `nn.Module` definitions, bridging regression concepts with classification tasks.

**Key Takeaway:** This module demonstrates how PyTorch streamlines model training, optimization, and evaluation, while scaling regression into multi-dimensional and classification contexts. It connects the mechanics of linear models with the broader landscape of supervised learning.

---

## 🚧 Status

By completing the first three modules, the course has covered:

- How PyTorch represents and processes data using **tensors**.
- How **gradients and automatic differentiation** enable training through backpropagation.
- How to implement and train **linear regression models** with gradient descent.
- How optimization strategies like **SGD, mini-batch processing, and optimizers** make training efficient.
- How to prevent overfitting using **validation and test splits**.
- How regression concepts extend to **multi-dimensional inputs/outputs** and evolve into **logistic regression for classification**.
