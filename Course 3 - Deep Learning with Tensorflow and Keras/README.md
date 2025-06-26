# 📚 Course 3 – Advanced Deep Learning with TensorFlow and Keras

Welcome to the course **Deep Learning with TensorFlow and Keras**, part of the **AI Engineer Professional Certificate by IBM**.

The course focuses on mastering model customization, handling complex data types (like images and sequences), and building architectures for classification, generation, and reinforcement learning.

Across six modules, the course explores custom training workflows, convolutional architectures, Transformer encoders, generative models for unsupervised learning, advanced optimization strategies, and reinforcement learning using Deep Q-Networks.

---

## 📌 Course Objectives

- Perform tensor operations and build linear/logistic regression models using TensorFlow and Keras.
- Understand and apply advanced Keras functionalities, including custom layers and models.
- Build deep convolutional networks with data augmentation and transfer learning.
- Develop Transformer models for sequential and textual data.
- Apply unsupervised learning techniques, including Autoencoders, Diffusion Models, and GANs.
- Implement reinforcement learning algorithms using Q-Learning and Deep Q-Networks.
- Train and optimize models with custom loops and hyperparameter tuning.
- Complete an end-to-end classification project integrating multiple deep learning concepts.

---

## 🧠 Key Concepts by Module

### 🔧 Module 1 – Advanced Keras Functionalities

- Implemented custom training loops using `GradientTape` for full control of model training logic.
- Used the Functional API and `Model` subclassing to construct reusable architectures.
- Built custom layers with the `Layer` class, defining behavior in `build()` and `call()`.
- Created custom callbacks and training metrics to manage training feedback and control flow.
- Combined all techniques in an end-to-end training pipeline.

---

### 🌀 Module 2 – Advanced CNNs in Keras

- Designed convolutional networks for image classification using Conv2D, ReLU, and pooling layers.
- Applied dropout to prevent overfitting and used batch normalization to stabilize training.
- Managed training with callbacks: model checkpointing and early stopping.
- Monitored training metrics with TensorBoard.
- Built a complete ConvNet architecture on the Fashion MNIST dataset.

---

### 🔁 Module 3 – Transformers in Keras

- Implemented a Transformer encoder architecture with multi-head self-attention and feedforward layers.
- Constructed positional encodings to retain sequence information in tokenized inputs.
- Applied masking techniques (padding and look-ahead) to guide attention behavior.
- Created a full Transformer model from scratch for classification tasks.
- Used `TextVectorization` to preprocess and embed text sequences for training.

---

### 🎨 Module 4 – Unsupervised Learning and Generative Models in Keras

- Trained autoencoders for dimensionality reduction and feature reconstruction tasks.
- Implemented a basic diffusion model using a U-Net-inspired architecture.
- Built and trained GANs for generating synthetic MNIST digits.
- Used visual inspection (clarity, coherence, diversity) and metrics (discriminator accuracy) to evaluate GAN outputs.
- Applied adversarial training techniques including label smoothing and normalization.

---

### ⚙️ Module 5 – Advanced Keras Techniques

- Developed custom training logic with `GradientTape` and manual metric tracking.
- Created specialized layers and custom `Callback` classes for dynamic control during training.
- Tuned model hyperparameters using Keras Tuner (RandomSearch) with parameterized model definitions.
- Applied model optimization techniques:
  - He weight initialization
  - Learning rate scheduling
  - Mixed precision training
  - TensorFlow Model Optimization Toolkit for pruning and quantization
- Trained smaller student models via knowledge distillation to match the outputs of large teacher networks.

---

### 🧠 Module 6 – Introduction to Reinforcement Learning with Keras

- Introduced reinforcement learning concepts: agents, environments, actions, states, and rewards.
- Implemented Q-learning using the Q-value function `Q(s, a)` and the Bellman equation.
- Built a neural Q-network in Keras to replace Q-tables for continuous state spaces.
- Integrated experience replay buffers and target networks to stabilize DQN training.
- Trained and evaluated DQNs in the CartPole environment using an epsilon-greedy policy and cumulative reward tracking.

---

## 🔑 What I Learned

- Built advanced model architectures using Keras’s low-level APIs and custom logic.
- Learned to train and evaluate CNNs, Transformers, autoencoders, GANs, and RL agents.
- Used advanced tooling for optimization, including mixed precision, quantization, and distillation.
- Applied reinforcement learning to practical environments using Deep Q-Networks and OpenAI Gym.

---
