# 🔢 Module 1 – Sections 1: Tensors 1D and 2D

These sections cover the foundational concepts of one-dimensional and two-dimensional tensors in PyTorch. Tensors serve as the primary data structures for representing and processing information in neural networks. The material explores how tensors are structured, accessed, transformed, and used in computation—establishing a baseline for working with models in PyTorch.

---

## 🧩 Conceptual Overview

Tensors in PyTorch represent data in structured, n-dimensional arrays. They form the inputs, outputs, and trainable parameters in a neural network.

- **1D tensors** represent linear sequences such as feature vectors, time series, or single rows from datasets.
- **2D tensors** represent tabular datasets and image-like matrices, where rows typically correspond to samples and columns to features.

Tensors generalize mathematical arrays and support operations aligned with linear algebra, enabling efficient data manipulation and integration with the broader Python ecosystem.

---

## 🧠 Structural and Analytical Insights

- Tensors contain values of a single data type (e.g., float, int, byte).
- Structural properties include:
  - **Rank**: Number of dimensions (e.g., 1 for vectors, 2 for matrices).
  - **Shape**: Size across dimensions (e.g., 100 rows × 1 column).
  - **Size**: Total number of elements.
- Tensors support reshaping and dimensional transformation, such as converting 1D tensors into column vectors to match model input requirements.

PyTorch tensors are fully compatible with NumPy arrays and Pandas series, allowing bidirectional conversion and memory sharing for high-performance workflows.

---

## 🧮 Indexing, Slicing, and Data Access

- **1D tensors** can be indexed and sliced like Python lists.
- **2D tensors** support two-dimensional indexing using row and column positions.
- Values can be accessed, updated, or extracted as sub-tensors for preprocessing or feature manipulation.

This behavior enables direct compatibility with Pythonic iteration, making tensors easy to work with in loops or conditional structures.

---

## ⚙️ Operations and Transformations

Tensors support a wide range of mathematical operations critical to neural network computation:

- **Element-wise operations**:
  - Addition
  - Scalar multiplication
  - Hadamard product (element-wise multiplication)
- **Dot product**: Measures vector similarity (1D).
- **Matrix multiplication**: Applies linear transformations (2D), used for computing outputs in neural layers.
- **Broadcasting**: Allows operations between tensors of different shapes by extending dimensions implicitly.

Mathematical functions (e.g., sine, mean, max) can be applied across tensor elements. PyTorch also supports generating evenly spaced tensors for simulation or plotting purposes.

---

## 📊 Visualization and Ecosystem Integration

- Tensors can be visualized by converting them to NumPy arrays and using libraries like Matplotlib.
- Signal generation, transformation inspection, and functional plots can be constructed for validation or presentation.
- Integration with Python tools supports efficient debugging, data inspection, and experiment tracking.

---

## ✅ Key Takeaways

- Tensors are core structures for building, training, and running neural networks in PyTorch.
- 1D tensors model simple data structures such as vectors or time series.
- 2D tensors are ideal for structured datasets and image inputs.
- Tensors can be reshaped, sliced, and transformed to align with model architecture requirements.
- Mathematical operations on tensors mirror linear algebra and support scalable, GPU-accelerated model training.
- Full interoperability with NumPy and Pandas allows seamless integration into existing data pipelines and visualization tools.
