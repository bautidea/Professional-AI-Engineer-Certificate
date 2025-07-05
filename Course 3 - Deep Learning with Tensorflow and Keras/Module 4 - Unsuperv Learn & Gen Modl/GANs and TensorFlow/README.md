# 🧠 Module 4 - Section 2: GANs and TensorFlow

This section focuses on two essential components of unsupervised learning: **Generative Adversarial Networks (GANs)** and **TensorFlow’s support for unsupervised learning workflows**. Learners implemented GAN architectures and explored how to leverage TensorFlow for clustering and dimensionality reduction.

---

## 🎨 Generative Adversarial Networks (GANs)

GANs are a powerful class of neural networks capable of generating synthetic data that closely resembles real data. Their adversarial architecture pits two models against each other:

- **Generator**: Creates synthetic data (e.g., images) from random noise.
- **Discriminator**: Evaluates whether a given image is real (from the dataset) or fake (from the generator).

These models are trained simultaneously. The generator improves to fool the discriminator, while the discriminator sharpens its ability to detect fakes. This adversarial loop continues until the generated outputs become indistinguishable from real data.

### 🧱 Key Components

- **Generator Model**:

  - Input: 100-dimensional noise vector.
  - Output: Synthetic image (e.g., 28x28 MNIST digit).
  - Structure: Dense layers with activations shaping noise into structured pixel distributions.

- **Discriminator Model**:

  - Input: Real or synthetic image.
  - Output: Probability score (real or fake).
  - Structure: Dense layers for binary classification.

- **GAN Composition**:
  - The generator and discriminator are stacked.
  - During training, the discriminator is frozen while updating the generator to ensure adversarial training remains stable.

---

## 🔁 GAN Training Workflow

Training proceeds in alternating steps:

1. **Train Discriminator**:

   - Sample real images from the dataset.
   - Generate synthetic images with the generator.
   - Update the discriminator to correctly classify real vs. fake images.

2. **Train Generator**:
   - Generate new images from noise.
   - Update generator weights to fool the discriminator into labeling them as real.

This loop continues for many epochs. Over time, both networks become more sophisticated, with the generator learning to produce high-quality images.

---

## 🧪 Evaluating the GAN

A robust evaluation involves both qualitative inspection and quantitative metrics:

### ✅ Qualitative Inspection

Use the `sample_images()` function to generate a grid of synthetic images at intervals during training.

Review generated samples for:

- **Clarity**: Sharp, defined images (no blurriness).
- **Coherence**: Images resemble real digits in shape and structure.
- **Diversity**: Wide variation among outputs; lack of diversity can signal mode collapse.

This human-in-the-loop review helps catch model degradation or early signs of failure.

### 📏 Quantitative Metrics

- **Discriminator Accuracy**:

  - Target value ~50%.
  - Indicates the generator is fooling the discriminator effectively.
  - Higher/lower values suggest imbalance or poor convergence.

- **Fréchet Inception Distance (FID)**:

  - Measures the similarity between distributions of generated vs. real images.
  - Lower scores indicate more realistic outputs.

- **Inception Score (IS)**:
  - Evaluates both the confidence and diversity of generated outputs.
  - High scores suggest varied and well-defined images.
  - Less useful for simple datasets like MNIST.

Combining visual inspection with metric tracking gives a full view of GAN performance.

---

## ⚙️ TensorFlow for Unsupervised Learning

TensorFlow offers extensive support for unsupervised learning tasks. In this section, learners explored:

### 🔍 Clustering with K-Means

1. **Preprocess the MNIST dataset**:

   - Normalize pixel values.
   - Flatten images.

2. **Apply K-Means Clustering**:
   - Cluster the images into 10 groups.
   - Visually inspect representative samples from each cluster.

Clustering reveals patterns in data and enables grouping based on similarity without labels.

---

### 📉 Dimensionality Reduction with Autoencoders

Autoencoders compress high-dimensional input into low-dimensional representations, then reconstruct the original input.

#### 🏗️ Model Architecture:

- **Input Layer**: Flattened MNIST images (784 features).
- **Encoder**: Reduces to a bottleneck latent space.
- **Decoder**: Reconstructs the original image.

#### 🧪 Visualization with t-SNE:

- After training, the 2D embeddings from the bottleneck are projected using t-SNE.
- Plotting these embeddings reveals how well the autoencoder separates and compresses class-like patterns.

---

## ✅ Key Takeaways

- GANs are adversarial models composed of a generator and discriminator, trained in opposition.
- A well-trained GAN can produce realistic data from pure noise.
- Evaluation of GANs requires both visual inspection and quantitative metrics like FID and discriminator accuracy.
- TensorFlow supports key unsupervised learning tasks such as clustering and dimensionality reduction through K-Means and autoencoders.
- Visual tools such as t-SNE plots help inspect model effectiveness in feature compression and class separation.
