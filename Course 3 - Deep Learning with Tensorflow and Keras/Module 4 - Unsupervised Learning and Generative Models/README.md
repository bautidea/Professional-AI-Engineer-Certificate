# 🧠 Module 4 – Unsupervised Learning and Generative Models in Keras

This module focuses on building unsupervised learning models using Keras and TensorFlow. It covers clustering, dimensionality reduction, autoencoders, diffusion models, and GANs for tasks such as data compression, image denoising, and synthetic data generation.

---

## 🔍 Unsupervised Learning Principles

Unsupervised learning involves training models on unlabeled data to uncover patterns or structure. Key categories include:

- **Clustering**: Grouping similar data points (e.g., K-Means).
- **Dimensionality Reduction**: Reducing feature space while preserving key information (e.g., Autoencoders, t-SNE).
- **Anomaly Detection**: Identifying data points that deviate from the norm.

Applications: customer segmentation, fraud detection, and image compression.

---

## 🧬 Autoencoders with Keras

Autoencoders are neural networks trained to recreate input data. They are useful for:

- Dimensionality reduction
- Data denoising
- Feature learning

### 🏗️ Architecture

- **Encoder**: Compresses input into a latent space
- **Bottleneck**: Captures the most informative features
- **Decoder**: Reconstructs the input from the latent representation

### ⚙️ Implementation

- Dataset: MNIST
- Loss Function: Binary crossentropy
- Optimizer: Adam
- Fine-Tuning: Unfreeze final layers and retrain

### 📈 Visualization

Used t-SNE to project the bottleneck representations into 2D space, revealing clear digit clusters.

---

## 🌫️ Diffusion Models

Diffusion models generate data by reversing a noise process. They iteratively refine a noisy sample until it becomes coherent.

### 🧠 Concept and Operation

- **Forward Process**: Adds noise over multiple time steps, degrading the input into random noise.
- **Reverse Process**: Trains a model to denoise data step-by-step, gradually reconstructing the original sample.

Inspired by physical diffusion, these models simulate how particles spread from high to low concentration.

### 🧰 Applications

- Image generation from noise
- Image denoising
- Data augmentation

### 🏗️ Implementation

- Architecture: Convolutional Neural Network
- Input: Noisy MNIST images
- Output: Denoised reconstructions
- Evaluation: Visual comparison of original, noisy, and denoised images
- Fine-Tuning: Retrained last layers for improved denoising quality

---

## 🎨 Generative Adversarial Networks (GANs)

GANs generate realistic synthetic data using adversarial training between two networks:

### 🔁 Architecture

- **Generator**: Produces images from random noise
- **Discriminator**: Distinguishes real from fake images

### ⚙️ Training Loop

1. Train discriminator on real and fake images
2. Train generator to fool the discriminator
3. Repeat until generator produces realistic samples

### 🧪 Evaluating the GAN

#### ✅ Qualitative (Visual) Assessment

- Use `sample_images()` to visualize output
- Look for:
  - **Clarity**: Sharp and clean images
  - **Coherence**: Recognizable digits (MNIST)
  - **Diversity**: Avoid mode collapse

#### 📏 Quantitative Metrics

- **Discriminator Accuracy**: ~50% suggests good balance
- **FID (Fréchet Inception Distance)**: Measures similarity to real images
- **IS (Inception Score)**: Assesses clarity and diversity (less suited for MNIST)

### 📊 Combined Evaluation Strategy

1. Visual inspection for immediate feedback
2. Metric tracking for objective validation
3. Loss monitoring to detect imbalance or instability

---

## ⚙️ TensorFlow for Unsupervised Learning

TensorFlow simplifies the implementation of unsupervised learning workflows.

### 🔍 Clustering (K-Means)

- Preprocessing: Normalize and reshape MNIST
- Apply K-Means to group images into 10 clusters
- Visualize representative images per cluster

### 📉 Dimensionality Reduction with Autoencoders

- Train encoder-decoder architecture
- Extract bottleneck features
- Use t-SNE to visualize compressed space

---

## ✅ Key Takeaways

- Unsupervised learning models find structure in unlabeled data, supporting clustering, compression, and anomaly detection.
- Autoencoders effectively reduce dimensionality and denoise images by learning compressed representations.
- Diffusion models simulate and reverse noise to generate high-quality synthetic data.
- GANs use adversarial training to generate realistic samples, evaluated using both visual and quantitative methods.
- TensorFlow provides robust support for building and visualizing unsupervised learning pipelines.
