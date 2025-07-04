# 🧠 Module 1 – Section 3: Derivatives in PyTorch

This section introduces the concept of differentiation in PyTorch and explains how gradients are computed automatically using the framework's internal graph system. It covers both single-variable and multivariable functions, and how derivative information is used in neural network training.

---

## 🧩 Conceptual Overview

Differentiation is a foundational concept in deep learning, enabling models to learn through gradient-based optimization. PyTorch provides built-in tools for computing derivatives using a dynamic computational graph called the backward graph.

When tensors are created with gradient tracking enabled, PyTorch automatically records all operations applied to them. This information is stored as metadata and used later to calculate gradients during backpropagation.

---

## ⚙️ Functional Insights

- A tensor marked for gradient tracking serves as the input to functions defined in terms of that tensor.
- Calling a backward pass on the output function triggers computation of the derivative, storing the result in the input tensor’s gradient field.
- This mechanism supports both simple scalar functions and multivariable expressions.

Each tensor maintains key attributes involved in gradient computation:

- **Data**: The raw numeric value.
- **Grad**: The computed derivative after backpropagation.
- **Grad_fn**: A reference to the operation that created the tensor.
- **Is_leaf**: Marks whether the tensor is a leaf in the backward graph.
- **Requires_grad**: Indicates whether PyTorch should track operations on the tensor.

---

## 🧮 Single and Partial Derivatives

### Single-variable Differentiation

- Derivatives are computed automatically from scalar-valued functions of one tensor.
- Once the function is defined and evaluated, a backward pass calculates and stores the gradient relative to the input.

### Partial Derivatives

- For functions of multiple variables, PyTorch computes the partial derivative with respect to each input independently.
- Each input tensor must be marked for gradient tracking.
- Gradients are retrieved from the respective tensor after a backward pass is executed on the function.

---

## ✅ Key Takeaways

- PyTorch builds a computational graph behind the scenes to automate differentiation.
- Both single-variable and multivariable gradients are supported through dynamic graph construction.
- Gradient information is stored directly in the input tensors and is accessible for analysis or optimization.
- Attributes like `grad`, `grad_fn`, `requires_grad`, and `is_leaf` provide insight into how tensors participate in gradient flows.
- Automatic differentiation is essential for training neural networks using techniques such as gradient descent.
