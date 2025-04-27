# 📚 Module 4 - Deep Learning Models

Welcome to Module 4 of the course **Introduction to Deep Learning and Neural Networks with Keras**.  
In this module, I explored key architectures in deep learning, understanding both **supervised** and **unsupervised** models, and built a strong foundation in **feature extraction**, **sequence modeling**, and **transformer-based architectures**.

---

## 📌 Topics Covered

---

### 1️⃣ Shallow vs Deep Neural Networks

- **Shallow Neural Networks** consist of 1–2 hidden layers and are suitable for structured data.
- **Deep Neural Networks** have 3+ hidden layers and are capable of handling complex, unstructured data like images and text.
- The modern success of deep learning is driven by:
  - Advancements like the ReLU activation function.
  - Availability of large datasets.
  - Computational power improvements (GPUs).

---

### 2️⃣ Convolutional Neural Networks (CNNs)

- CNNs specialize in processing grid-like data (e.g., images).
- Key components:
  - **Input Layer** for 3D data (height × width × channels).
  - **Convolutional Layers** extract local features with filters.
  - **ReLU Activation** introduces non-linearity.
  - **Pooling Layers** reduce spatial dimensions and add spatial invariance.
  - **Fully Connected Layers** interpret extracted features.
- CNNs are widely used in image classification, object detection, and segmentation tasks.

---

### 3️⃣ Recurrent Neural Networks (RNNs) and LSTMs

- RNNs handle sequential data by introducing **memory** through feedback loops.
- Each output depends not just on the current input but also on the previous hidden state.
- **LSTM networks** overcome the vanishing gradient problem, enabling learning over longer sequences.
- Applications include:
  - Language modeling
  - Stock price prediction
  - Audio and handwriting generation

---

### 4️⃣ Transformers

- Transformers revolutionized sequence processing by using **attention mechanisms** instead of recurrence.
- **Self-Attention Mechanism**:
  - Each token attends to every other token to build contextualized embeddings.
- **Cross-Attention Mechanism** (Text-to-Image Generation):
  - Enables models like DALL·E to generate images based on textual prompts by linking textual queries to visual representations.
- Transformers are fully parallelizable, resulting in faster training compared to RNNs.
- Transformers now power cutting-edge AI systems like ChatGPT, BERT, and DALL·E.

---

### 5️⃣ Autoencoders

- **Autoencoders** are unsupervised neural networks that learn to compress and then reconstruct their inputs.
- Key architecture:
  - **Encoder**: Compresses input into a latent representation.
  - **Decoder**: Reconstructs the input from the latent space.
- Applications include:
  - Data denoising
  - Dimensionality reduction
  - Feature extraction
- **Restricted Boltzmann Machines (RBMs)** are a specialized type of autoencoder used for handling imbalanced datasets and imputing missing values.

---

### 6️⃣ Using Pre-trained Models

- **Pre-trained models** (e.g., VGG16, ResNet) trained on large datasets like ImageNet can be used as **feature extractors** without retraining.
- Benefits:
  - Save training time and resources.
  - Achieve high performance with limited data.
- **Fine-Tuning**:
  - Unfreeze and retrain a few top layers of the pre-trained model.
  - Allows adaptation to new tasks where the data distribution slightly differs from the original dataset.

---

## ✅ Key Takeaways

- Deep learning architectures vary in depth and specialization, from shallow fully connected networks to CNNs, RNNs, and transformers.
- CNNs and RNNs handle spatial and temporal data, respectively, while transformers model long-range dependencies more efficiently through self-attention.
- Autoencoders provide powerful unsupervised learning capabilities for feature compression and reconstruction.
- Transfer learning and fine-tuning allow leveraging large pre-trained models, making deep learning accessible even with limited data and computation.
