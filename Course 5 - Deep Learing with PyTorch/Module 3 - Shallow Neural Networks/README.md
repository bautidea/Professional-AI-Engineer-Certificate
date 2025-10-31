# 🧠 Module 3 — Shallow Neural Networks

## 📘 Introduction

This module introduces the foundations of **neural network architecture**, focusing on **shallow neural networks** with one hidden layer. It explores how networks approximate nonlinear relationships, how increasing model complexity impacts performance, and how gradient-based optimization enables learning.

The module also covers **activation functions**, **backpropagation**, and **common optimization challenges** such as **vanishing gradients**, **overfitting**, and **underfitting** — building the conceptual basis for deeper architectures in later modules.

---

## 🧩 Topics and Concepts

### 🔹 Understanding Neural Networks

A **neural network** is a mathematical function composed of **linear transformations** and **nonlinear activations**, capable of approximating complex, nonlinear relationships.

Each neuron performs two key operations:

- A **linear transformation** of inputs through weights and biases.
- A **nonlinear activation** that introduces flexibility and allows the network to learn curved decision boundaries.

Even a shallow network with one hidden layer can model nonlinear separations that simple linear classifiers cannot achieve. The hidden layer’s activations form intermediate representations of data, while the output layer combines these activations to produce predictions.

In PyTorch, networks can be constructed using:

- **`nn.Module`**, for full customization of architecture and forward logic.
- **`nn.Sequential`**, for simpler, layer-by-layer model definitions.

Both integrate seamlessly with PyTorch’s **automatic differentiation** and **gradient-based training** workflows.

### 🔹 Model Capacity and Hidden Neurons

The **number of neurons** in a hidden layer determines the network’s representational power.

- With **too few neurons**, the model cannot capture complex relationships (underfitting).
- With **more neurons**, the network learns richer patterns and finer decision boundaries (higher flexibility).

Each hidden neuron generates its own nonlinear transformation of the input, and the weighted combination of these outputs allows the network to form **smooth, composite functions**.

However, increasing model size also increases the risk of **overfitting** and demands careful regularization and optimization strategies.

### 🔹 Multi-Dimensional Inputs

When neural networks process **multi-feature inputs**, each feature receives its own weight, enabling the model to construct multidimensional decision surfaces.  
In two or more dimensions, hidden neurons produce curved, nonlinear boundaries that separate classes that linear models cannot distinguish.

This section also introduces two key concepts:

- **Overfitting:** The model is too complex and learns noise, reducing generalization.
- **Underfitting:** The model is too simple and fails to capture essential patterns.

Balancing model complexity involves using **validation sets**, **regularization techniques**, and **data augmentation** to ensure optimal generalization.

### 🔹 Multi-Class Neural Networks

Neural networks can be extended from binary to **multi-class classification** by assigning one output neuron per class.  
Each output neuron computes a score (logit) for its class, and the class with the highest score is selected as the prediction.

In PyTorch, **`nn.CrossEntropyLoss()`** integrates both **Softmax** normalization and **log-likelihood** loss, eliminating the need for an explicit Softmax activation in the output layer.

This framework generalizes the concept of logistic regression to multiple categories, allowing a single network to distinguish between many classes simultaneously.

### 🔹 Backpropagation and the Vanishing Gradient

**Backpropagation** is the core algorithm that enables neural networks to learn. It applies the **chain rule** of calculus to compute how changes in each weight affect the loss function, propagating error signals backward through the network.

This efficient process reuses partial derivatives across layers, drastically reducing computational cost.

However, deep networks face the **vanishing gradient problem**:

- When using activation functions like **sigmoid** or **tanh**, gradients shrink exponentially as they propagate backward.
- Early layers receive minimal updates, causing training to stagnate.

Solutions include using **ReLU activations**, **proper weight initialization**, **batch normalization**, and **adaptive optimizers** like **Adam** or **RMSProp**.

### 🔹 Activation Functions

Activation functions define how signals are transformed between layers. They determine not only the network’s expressive capacity but also the stability of gradient propagation during training.

#### **Sigmoid**

- Maps values between 0 and 1.
- Useful for probabilistic outputs.
- Suffers from **severe vanishing gradients** in deep networks.

#### **Tanh**

- Zero-centered, mapping values between -1 and 1.
- Improves training stability but still experiences **gradient saturation**.

#### **ReLU**

- Outputs 0 for negative inputs and the input itself for positive ones.
- Partially solves the vanishing gradient problem.
- Enables **faster convergence** and **sparse activations**.

Empirically, **ReLU** and **Tanh** achieve lower training loss and higher validation accuracy than Sigmoid, making them preferred choices for most architectures.

---

## ✅ Takeaways

- **Neural Networks as Function Approximators:** Even shallow networks with one hidden layer can model complex nonlinear relationships.
- **Model Flexibility:** Increasing neurons improves expressiveness but raises the risk of overfitting, requiring validation and regularization.
- **Multidimensional Learning:** Networks can process multiple input features, forming nonlinear decision surfaces that generalize across spaces.
- **Gradient-Based Learning:** Backpropagation efficiently computes gradients using the chain rule, enabling optimization through gradient descent.
- **Vanishing Gradient Challenge:** Deep models may struggle with small gradients, mitigated by ReLU activations and proper initialization.
- **Activation Function Design:** The choice of activation directly influences convergence speed, gradient flow, and model performance.
