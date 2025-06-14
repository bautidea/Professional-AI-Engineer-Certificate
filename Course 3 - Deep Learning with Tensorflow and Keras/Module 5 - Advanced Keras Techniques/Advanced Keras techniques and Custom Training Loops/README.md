# 🚀 Module 5 – Section 1: Advanced Keras Techniques and Custom Training Loops

This section focuses on using Keras beyond its high-level defaults. It covers techniques that allow full control over how models are built, trained, and optimized — crucial when working on custom architectures or deploying models in real-world settings.

---

### 🔄 Custom Training Loops

Instead of relying on `model.fit()`, this section shows how to write training logic manually using `tf.GradientTape`. This is useful when:

- The training process needs to be customized (e.g., unusual loss functions or training strategies).
- You want detailed control over every training step, including gradient computation and weight updates.
- You're debugging training behavior or integrating with multiple models (like GANs).

We walk through how to structure a full training loop: dataset batching, forward pass, loss calculation, gradient computation, and optimizer application.

---

### 🧱 Custom Layers

We implement both a custom Dense layer and explore Lambda layers:

- **Custom Dense Layer**: Subclassing `Layer` to define weights and logic manually. Useful for adding constraints or operations not supported by built-in layers.
- **Lambda Layer**: Quick way to apply custom tensor operations inline. Great for fast experimentation when no learnable parameters are needed.

This shows how to go beyond off-the-shelf layers and extend Keras with your own building blocks.

---

### 🔔 Advanced Callbacks

Built-in and custom callbacks are explored for better training automation:

- **Built-in**: `EarlyStopping`, `ModelCheckpoint`, and others to monitor training and react dynamically.
- **Custom**: Subclassing `Callback` to implement behaviors like logging custom metrics, triggering actions on specific conditions, or adding live feedback.

These tools are especially useful in research workflows or when managing complex training processes.

---

### ⚡ Mixed Precision Training

We enable TensorFlow’s mixed precision policy to speed up training using float16 operations. This improves performance on compatible hardware (like modern GPUs) while keeping accuracy stable.

It’s a simple config change, but it significantly reduces memory usage and speeds up training — especially for large models.

---

### 🛠️ TensorFlow Model Optimization Toolkit

We introduce model compression techniques that are essential for deployment:

- **Quantization**: Reduces model size and inference time by converting weights to lower precision (e.g., int8).
- **Pruning**: Removes unnecessary weights during training, creating smaller, faster models without large accuracy drops.

These techniques are key when deploying models on mobile devices or constrained environments.

---

This section shows hands-on experience with:

- Writing custom training logic using TensorFlow low-level APIs.
- Extending Keras with custom and Lambda layers.
- Automating model behavior with custom callbacks.
- Improving performance using mixed precision.
- Preparing models for deployment through quantization and pruning.
