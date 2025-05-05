# 🔧 Module 1 - Advanced Keras Functionalities

## Section: Custom Layers & TensorFlow 2.x Overview

This section focuses on extending model capabilities through **custom Keras layers** and understanding the modern architecture and tools provided by **TensorFlow 2.x**. Together, these topics form the foundation for building advanced, flexible, and production-ready deep learning systems.

---

## 📌 Topics Covered

---

### 🧱 Creating Custom Layers in Keras

Custom layers enable you to define new behavior inside a neural network model — essential for research, optimization, or domain-specific logic.

#### 🔹 Why Use Custom Layers?

- Go beyond standard layers like `Dense`, `Conv2D`, or `LSTM`.
- Encapsulate reusable logic and improve code clarity.
- Tailor performance or memory usage for specific constraints.
- Implement novel architectures for experimentation and innovation.

#### 🔹 Lifecycle of a Custom Layer

To define a custom layer, subclass `tensorflow.keras.layers.Layer` and implement:

- **`__init__()`**: Set layer-level configurations (e.g., units, activation).
- **`build(input_shape)`**: Create and initialize weights with `self.add_weight(...)`. This runs only once.
- **`call(inputs)`**: Define the forward pass computation.

#### 🔹 Example Use Case

Implement a custom dense layer with ReLU activation:

```python
class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
```

This layer can be plugged into any Keras model:

```python
model = tf.keras.Sequential([
    CustomDense(64),
    CustomDense(10)
])
```

---

## ⚙️ Overview of TensorFlow 2.x

TensorFlow 2.x is a complete platform for building, training, and deploying machine learning models across various environments — from cloud and servers to mobile and web.

### 🔹 Key Features

#### 🧠 Eager Execution

Runs operations **immediately** as they are called — no need for static computation graphs.

**Benefits:**

- **Debugging**: Immediate feedback makes errors easier to trace.
- **Readability**: More intuitive, Pythonic code.
- **Flexibility**: Supports dynamic models and runtime behaviors.

#### 🛠 Keras as High-Level API

TensorFlow 2.x integrates Keras for defining, training, and deploying models using a **modular and user-friendly syntax**.

**Benefits:**

- Concise and clean model definitions.
- Easy combination of layers, losses, and optimizers.
- Backed by extensive documentation and community support.

#### 🌍 Multi-Platform Support

TensorFlow models can be deployed across:

- Servers and cloud (with CPU, GPU, TPU support)
- Mobile devices (via **TensorFlow Lite**)
- Web applications (via **TensorFlow.js**)
- Embedded systems and edge devices

### 🌐 TensorFlow Ecosystem Components

| Tool                | Description                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| **TensorFlow Lite** | Lightweight runtime for mobile and embedded inference                    |
| **TensorFlow.js**   | Enables training and inference in browsers and Node.js environments      |
| **TFX**             | TensorFlow Extended: Manages production ML pipelines and workflows       |
| **TensorFlow Hub**  | Repository of reusable pre-trained models and components                 |
| **TensorBoard**     | Visualization toolkit for monitoring training, metrics, and architecture |

---

### ✅ Key Takeaways

- **Custom Keras layers** provide the flexibility to define domain-specific operations and encapsulate reusable logic inside models.
- **TensorFlow 2.x** simplifies model development with eager execution, while enabling robust deployment through integrated tools and multi-platform support.
- Ecosystem tools like **TFLite**, **TensorFlow.js**, **TFX**, **TF Hub**, and **TensorBoard** empower cross-platform ML, visualization, and scalable pipeline management.
