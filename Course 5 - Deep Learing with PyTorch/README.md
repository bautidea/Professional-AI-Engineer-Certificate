# 🧠 Deep Learning with PyTorch

## 📖 Introduction

This course introduces the principles and implementation of deep learning using **PyTorch**, an open-source framework designed for building and training neural networks. It builds a bridge between classical machine learning and modern deep learning architectures, guiding through practical model implementation, optimization, and evaluation.

The course progresses from the fundamentals of regression and classification into multi-layer architectures and convolutional networks. Each module integrates theoretical understanding with practical development using the **`torch`** and **`torch.nn`** libraries, emphasizing hands-on experience through structured projects and model training exercises.

---

## 🧩 Topics and Concepts

### 🔹 Module 1: Logistic Regression and Cross-Entropy Loss

The first module establishes the foundation of deep learning by revisiting **logistic regression** and analyzing the limitations of **mean squared error (MSE)** for classification tasks. It introduces the **cross-entropy loss** function, derived from **maximum likelihood estimation (MLE)**, as a more suitable alternative for probabilistic classification.  
Through PyTorch, models are implemented using the **`nn.ModuleList`** structure, enabling modular design and direct gradient-based optimization. The focus lies on understanding how gradient descent updates parameters through the loss surface and how PyTorch automates this process with **autograd**.

### 🔹 Module 2: Softmax Regression

Building upon logistic regression, this module expands the framework to **multiclass classification** using **Softmax regression**. It explores how **Softmax** normalizes linear outputs into probabilities and how **Argmax** is used for class prediction.  
The implementation introduces the creation of custom **Softmax modules** using the **`nn.Module`** package, demonstrating the internal mechanics of forward propagation and parameter handling.  
Concepts such as **categorical cross-entropy**, **one-hot encoding**, and **decision boundaries** are integrated to solidify understanding of multiclass learning within PyTorch.

### 🔹 Module 3: Shallow Neural Networks

This module transitions from linear models to **feedforward neural networks** with a single hidden layer. Using **`nn.Module`** and **`nn.Sequential`**, models are constructed with greater flexibility, enabling nonlinear transformations through activation functions such as **Sigmoid**, **Tanh**, and **ReLU**.  
Core topics include **forward and backward propagation**, **gradient computation**, and **vanishing gradients**. The module also explores **overfitting** and **underfitting**, emphasizing the importance of network depth, regularization, and data balance.  
By the end, learners understand how shallow networks approximate nonlinear functions and how hidden layers improve representational capacity.

### 🔹 Module 4: Deep Networks

Expanding to deeper architectures, this module focuses on **deep neural networks** implemented through PyTorch’s modular components. It introduces **dropout layers** for regularization, **batch normalization** for stabilizing learning, and various **weight initialization techniques** to improve training convergence.  
Optimization strategies such as **stochastic gradient descent (SGD)** and **adaptive learning methods** are discussed alongside best practices for designing and debugging deeper models. The module demonstrates how each initialization strategy influences gradient flow, model performance, and convergence speed.

### 🔹 Module 5: Convolutional Neural Networks (CNNs)

This module explores **convolutional operations** for image data, detailing how local receptive fields and shared weights enable efficient feature extraction.  
Key components include **convolutional layers**, **pooling**, **activation functions**, and **fully connected layers**. Learners study how to compute **activation map sizes**, manage **multiple input and output channels**, and use **max pooling** to reduce spatial dimensions.  
Practical focus is placed on GPU acceleration with **CUDA** and on building scalable models such as **ResNet18**, introducing **residual learning** as a method for training deeper CNNs without vanishing gradient issues.

### 🔹 Module 6: Final Project — CNN on MNIST

The course culminates in a **hands-on project** applying all learned techniques to construct a **Convolutional Neural Network (CNN)** using PyTorch for image classification on the **MNIST dataset**.  
The project integrates:

- **Softmax regression** for output classification.
- **Layer stacking** for deep model construction.
- **Regularization** through dropout and batch normalization.
- **Performance evaluation** through accuracy metrics and visualization.

This project consolidates the full pipeline of deep learning development — from data preprocessing and model design to training, validation, and deployment readiness — producing a portfolio-quality implementation to showcase PyTorch proficiency.

---

## ✅ Key Takeaways

- **PyTorch Fundamentals:** Build and train deep learning models using `torch`, `nn.Module`, and `optim` tools.
- **Regression to Neural Networks:** Transition from linear models to deep, nonlinear architectures through structured progression.
- **Optimization and Regularization:** Apply techniques like dropout, batch normalization, and weight initialization to stabilize training.
- **Convolutional Architectures:** Design, train, and optimize CNNs with GPU acceleration for image data.
- **Portfolio Integration:** The final MNIST project provides a deployable artifact demonstrating applied understanding of PyTorch-based deep learning.
