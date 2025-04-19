# 🧠 Deep Learning with Keras - Module 3: Keras and Deep Learning Libraries

## 📖 Overview

Module 3 introduces the libraries behind deep learning and explores how to use the Keras library to build both **regression** and **classification** models. It covers the major deep learning frameworks, their trade-offs, and then walks through how to use Keras to create, train, and evaluate predictive models.

---

## 📌 Topics Covered

---

### 1️⃣ Deep Learning Libraries

Before building models, it’s important to understand the tools that support them. The most commonly used deep learning libraries are:

#### 🛠 TensorFlow

- Developed by **Google**
- Most used in **production environments**
- Large community, robust tooling
- **Steep learning curve** at low-level APIs

#### 🛠 PyTorch

- Developed by **Meta (Facebook)**
- Preferred in **academic research**
- Native to Python, flexible and dynamic
- Great GPU support, but also has a **learning curve**

#### 🛠 Keras

- High-level API built on top of **TensorFlow**
- Ideal for **beginners** and **rapid prototyping**
- Easy to use and understand
- Abstracts low-level details

| Feature               | TensorFlow | PyTorch   | Keras          |
| --------------------- | ---------- | --------- | -------------- |
| Developer             | Google     | Meta (FB) | Google         |
| Released              | 2015       | 2016      | High-level API |
| Learning Curve        | Steep      | Moderate  | Easy           |
| Ideal Use Case        | Production | Research  | Prototyping    |
| Control/Customization | High       | High      | Moderate       |
| Backend Dependency    | Native     | Native    | TensorFlow     |

✅ Keras is a great starting point to learn deep learning and build models with minimal code.

---

### 2️⃣ Regression Models with Keras

Regression tasks aim to predict a **continuous value** (e.g., temperature, price, score). Keras makes this process easy and efficient.

#### 🧱 Model Architecture

- Input layer: One neuron per feature
- Hidden layers: Typically use `ReLU` activation
- Output layer: One neuron with **no activation**

```python
from keras.models import Sequential
from keras.layers import Dense, Input

model = Sequential()
model.add(Input(shape=(predictors.shape[1],)))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
```

#### ⚙️ Compilation

- Loss function: MSE (measures average squared error)
- Optimizer: Adam (adaptive learning rate)

```python
model.compile(optimizer='adam', loss='mean_squared_error')
```

#### 🏋️‍♂️ Training & 🔮 Prediction

```python
model.fit(predictors, target, epochs=50, verbose=1)
predictions = model.predict(new_data)
```

---

### 3️⃣ Classification Models with Keras

Classification tasks assign each input to one of multiple discrete classes (e.g., types of animals, user categories, product segments).

🔁 What’s Different from Regression?

- The target column must be transformed using one-hot encoding using to_categorical().
- The output layer must use softmax activation.
- The loss function becomes categorical_crossentropy.
- Evaluation is done using accuracy.

#### 🧪 Target Encoding

Use one-hot encoding on class labels before training:

```python
from keras.utils import to_categorical
target_encoded = to_categorical(target)
```

#### 🧱 Model Architecture

- Output layer has one neuron per class
- Use softmax activation to output probabilities

```python
num_classes = target_encoded.shape[1]

model = Sequential()
model.add(Input(shape=(predictors.shape[1],)))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
```

#### ⚙️ Compilation

- Loss function: Categorical crossentropy (compares probability distributions)

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

#### 🏋️‍♂️ Training & 🔮 Prediction

```python
model.fit(predictors, target, epochs=50, verbose=1)
predictions = model.predict(new_data)
```

- Convert predictions to class labels:

```python
import numpy as np
predicted_classes = np.argmax(predictions, axis=1)
```

✅ Softmax ensures that all class probabilities sum to 1. The class with the highest probability is selected as the model's prediction

## ✅ Key Takeaways

Keras is a powerful, beginner-friendly tool for deep learning, enabling you to build both regression and classification models easily.

In regression, you use:

A single output neuron

No activation in the output

MSE as the loss

In classification, you use:

One-hot encoded labels

One output neuron per class

Softmax activation

Categorical crossentropy as the loss

Both types of models use the same Sequential API, Dense layers, and fit() method.

Keras allows fast experimentation and serves as a great entry point into deep learning, while still being flexible enough for more advanced applications.
