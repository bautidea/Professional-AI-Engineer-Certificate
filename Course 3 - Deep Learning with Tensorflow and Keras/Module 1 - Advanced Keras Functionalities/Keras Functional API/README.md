# 🧠 Module 1 - Advanced Keras Functionalities

## Section: Advanced Keras Functional API

This section focuses on leveraging the **Functional API** and the **Model Subclassing API** in Keras — two powerful tools for building **complex**, **multi-branch**, and **customizable neural network architectures** beyond the capabilities of the Sequential API.

---

## 📌 Topics Covered

### 🔷 Functional API in Keras

- Enables creation of **non-linear model topologies** (e.g., multi-branch networks, skip connections).
- Supports models with:
  - **Multiple inputs and/or outputs**
  - **Shared layers**
  - **Custom connection flows**

### 🔍 Functional API vs Sequential API

| Feature               | Sequential API            | Functional API                            |
| --------------------- | ------------------------- | ----------------------------------------- |
| Architecture          | Linear stack of layers    | Arbitrary layer graph                     |
| Flexibility           | Low                       | High (multi-input/output, branching)      |
| Use Case              | Simple feedforward models | Complex models (e.g., multitask learning) |
| Reusability of Layers | Limited                   | High — supports shared layers             |

### 🛠 Functional API Features and Scenarios

- **Multi-Input Models**: Design architectures where separate branches process different input sources before merging.
- **Shared Layers**: Apply the same layer instance across different parts of the model (e.g., Siamese networks).
- **Explicit Graph Construction**: Define every component (inputs, connections, outputs) clearly for full control and visibility.
- **Branching Architectures**: Design flexible workflows that mimic human decision trees or processing pipelines.

### 🔧 Functional API Code Patterns

- Define multiple `Input()` layers with distinct shapes.
- Connect layers manually using tensor chaining:  
  `x = Dense(...)(input_tensor)`
- Merge branches with tools like `Concatenate()` or `Add()`.
- Construct model with:  
  `Model(inputs=[...], outputs=[...])`

---

## 🔹 Model Subclassing API

- Offers the **highest level of flexibility** by allowing models to be defined using custom Python classes.
- Use cases:
  - **Dynamic architectures**
  - **Custom training loops**
  - **Experimental or research models**

### 🧱 Structure

- Define layers in `__init__()`
- Implement the forward pass in `call()`
- No static graph constraints — supports full Python control flow

### ⚙️ Custom Training with `tf.GradientTape`

- Enables low-level training control:
  - Compute loss manually
  - Track gradients
  - Apply updates with optimizers
- Useful when `.fit()` is not flexible enough (e.g., reinforcement learning, masked attention)

---

## ✅ Key Takeaways

- The **Functional API** is essential for building models involving **shared weights**, **multiple data paths**, or **hybrid outputs**.
- **Model Subclassing** unlocks dynamic behaviors, ideal for **non-standard architectures** or **fine-grained training logic**.
- These tools are foundational in modern AI workflows where **customization, transparency, and flexibility** are required.
