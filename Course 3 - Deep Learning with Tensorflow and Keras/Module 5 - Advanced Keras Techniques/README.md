# 🧠 Module 5 – Advanced Keras Techniques

This module explores advanced capabilities in TensorFlow and Keras that go beyond the high-level APIs. The goal is to enable full control over the training process, customize model behavior, and optimize performance across training and deployment environments.

You’ll learn to define training logic manually, create reusable model components, tune hyperparameters efficiently, and leverage TensorFlow's optimization tools for faster, smaller, and more accurate models.

---

## 🔄 Custom Training Loops

Keras provides a convenient `.fit()` method, but sometimes, you need full control over how a model learns. Custom training loops give you that control.

- Implement training loops manually using `tf.GradientTape`.
- Define your own loss calculations, optimization steps, and logging.
- Useful for research workflows, advanced metrics, and custom algorithms.

---

## 🧱 Custom Layers

Custom layers allow you to extend model functionality by subclassing the base `Layer` class.

- Use the `build()` method to define weights and biases.
- Use the `call()` method to control forward-pass logic.
- Ideal for adding custom activations or reusable functional blocks not available in Keras by default.

---

## 📟 Custom Callbacks

Callbacks let you react to events during training—like the end of an epoch or batch.

- Subclass the `Callback` class to log metrics, trigger early stopping, or save model checkpoints.
- Override methods for fine-grained control.
- Great for debugging, real-time monitoring, or automating training conditions.

---

## 🧠 Hyperparameter Tuning with Keras Tuner

Hyperparameters—like learning rate, number of layers, or batch size—can make or break model performance. Keras Tuner automates this search.

- Define tunable parameters with `hp.Int()` or `hp.Float()`.
- Use search strategies like Random Search, Hyperband, or Bayesian Optimization.
- Evaluate and retrain models using the best-found configuration.

---

## ⚙️ Model Optimization Techniques

Improving model performance isn't just about higher accuracy. It’s also about how efficiently the model trains and runs. This section dives into common and advanced optimization methods:

### ⚖️ Weight Initialization

- He Initialization (for ReLU) and Xavier Initialization (for tanh/sigmoid) help prevent training instability due to vanishing/exploding gradients.

### 📉 Learning Rate Scheduling

- Adjusts learning rate during training (e.g., constant for 10 epochs, then exponential decay).
- Helps the model converge faster and fine-tune performance in later epochs.

### 🧰 Post-Training Optimization

- **Batch Normalization**: Normalizes layer inputs to stabilize training.
- **Mixed Precision Training**: Uses both 16-bit and 32-bit floats to reduce memory and boost speed—especially on modern GPUs.
- **Model Pruning**: Removes unimportant weights, reducing model size with minimal accuracy loss.
- **Quantization**: Compresses weights to lower precision (e.g., int8) for edge deployment.

---

## 🚀 TensorFlow-Specific Optimizations

TensorFlow offers built-in support for optimizing models for deployment and scalability.

### 🧪 Knowledge Distillation

- Transfer knowledge from a large "teacher" model to a smaller "student" model.
- Trains the student using softened outputs (logits) from the teacher.
- Enables smaller models to perform well on limited-resource devices.

---

## 🔑 Key Concepts Recap

- Custom training loops give you full control using `tf.GradientTape`.
- Custom layers and callbacks help create flexible and extensible training setups.
- Keras Tuner automates the search for optimal hyperparameters.
- Weight initialization and learning rate scheduling boost training performance.
- TensorFlow Model Optimization Toolkit (TF MOT) enables post-training compression via pruning, quantization, and mixed precision.
- Knowledge distillation lets small models replicate high-accuracy behavior from larger ones.
