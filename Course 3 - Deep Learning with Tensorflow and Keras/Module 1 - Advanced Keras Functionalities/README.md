# 🧠 Deep Learning with TensorFlow and Keras

## Module 1: Advanced Keras Functionalities

This module explores advanced features in Keras and TensorFlow 2.x for building flexible, reusable, and production-ready deep learning models. It covers key APIs and architecture patterns that enable dynamic workflows, multi-input/output models, custom operations, and scalable deployment across platforms.

---

## 📌 Topics Covered

---

### 🔷 Keras Functional API

The Functional API enables the construction of **non-linear model architectures** using a graph-based structure. Unlike the Sequential API, it supports:

- **Multiple inputs and outputs**
- **Shared layers** (e.g., Siamese networks)
- **Complex data flows** (e.g., branching, merging, skip connections)

#### Key Features:

- Build custom, explicit computation graphs
- Reuse layers across inputs or model branches
- Visual clarity and better debugging via named graph components

#### Common Use Cases:

- Multi-task learning
- Multi-modal inputs (e.g., text + image)
- Architectures like ResNets, U-Nets, Siamese networks

#### Core Syntax:

```python
input1 = Input(shape=(...))
x = Dense(128, activation='relu')(input1)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=input1, outputs=output)
```

---

## 🧩 Model Subclassing API

The **Model Subclassing API** provides maximum control and flexibility for building custom or dynamic models in Keras.

### 🔹 Key Characteristics

- Models are Python classes that inherit from `tf.keras.Model`
- Layers are defined inside the `__init__()` constructor
- The forward pass is implemented in the `call()` method

### ✅ Benefits

- Supports **dynamic graph construction** (e.g., runtime conditionals, loops, or input-dependent behaviors)
- Ideal for:
  - Custom architectures
  - Reinforcement learning
  - Experimental workflows
  - Fine-grained training logic

### 💡 Example

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.d1(inputs)
        return self.d2(x)
```

---

## 🧱 Custom Layers in Keras

Custom layers allow developers to define **new behavior** within neural networks, offering complete control over **weight creation** and **forward computation logic**.

### 🔹 Use Cases

- Prototyping **new research concepts**
- Embedding **domain-specific logic**
- Encapsulating **reusable transformation blocks**

### 🔧 Structure of a Custom Layer

- `__init__()` – Configure layer parameters
- `build(input_shape)` – Create weights (called once during first input)
- `call(inputs)` – Define the forward computation logic

### 💡 Example

```python
class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros",
                                 trainable=True)

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
```

---

## ⚙️ TensorFlow 2.x Overview

TensorFlow 2.x is a modern machine learning framework designed to simplify both **research workflows** and **production deployment**.

### 🔹 Eager Execution

- Operations execute immediately, like native Python code
- Greatly improves **debuggability**
- Enables **interactive experimentation** (ideal for notebooks and prototyping)

### 🔹 Keras Integration

- Unified API for model building: `Sequential`, `Functional`, and `Subclassing`
- Offers concise, modular syntax for defining deep learning workflows
- Backed by a rich ecosystem and extensive official documentation

### 🔹 Multi-Platform Support

TensorFlow 2.x models can be deployed across:

- **Cloud and servers** (CPU, GPU, TPU)
- **Mobile devices** (via TensorFlow Lite)
- **Web applications** (via TensorFlow.js)
- **Embedded and edge devices**

## 🌐 TensorFlow Ecosystem Components

| Tool                | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| **TensorFlow Lite** | Lightweight runtime for mobile and embedded inference                       |
| **TensorFlow.js**   | Enables training and inference in browsers and Node.js                      |
| **TFX**             | TensorFlow Extended: Production-grade end-to-end ML pipeline platform       |
| **TensorFlow Hub**  | Repository of reusable pre-trained models and modules                       |
| **TensorBoard**     | Visualization toolkit for metrics, graphs, histograms, and model inspection |

---

## ✅ Key Takeaways

- **Functional API** enables high-flexibility modeling with clear, modular graph structures.
- **Model Subclassing** supports fully customized and dynamic architectures.
- **Custom Layers** provide total control over computation and parameter management.
- **TensorFlow 2.x** streamlines model development through:
  - Eager execution
  - Keras integration
  - Cross-platform deployment
  - A powerful and extensible ML ecosystem
