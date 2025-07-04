# 🧮 Module 1 – Section 2: Tensors 2D

This section focuses on the use of two-dimensional tensors in PyTorch. These tensors serve as a core structure for representing data in matrix form, enabling operations that are foundational in both machine learning workflows and image processing.

---

## 🧩 Conceptual Overview

Two-dimensional tensors represent data using rows and columns, forming a rectangular matrix. This structure is ideal for modeling:

- **Tabular datasets**, where each row corresponds to a sample and each column represents a feature or attribute.
- **Grayscale images**, where pixel intensities form a 2D grid with values typically ranging from 0 to 255.

The concept of dimensionality is extended beyond two axes. Color images, for example, are handled as three-dimensional tensors, with each color channel occupying its own 2D slice. More complex models may involve four or more dimensions.

---

## 🧠 Structural and Analytical Insights

- Tensors are built from nested sequences where each inner sequence forms a row in the 2D layout.
- Key structural attributes include:
  - **Rank** (number of dimensions)
  - **Shape** (number of rows and columns)
  - **Size** (total number of elements)

Understanding these characteristics is essential for preparing data for neural network inputs and aligning tensor shapes across layers.

- Indexing is performed using two indices: one for rows and one for columns. This allows precise access to any single value or subregion of a tensor.
- Slicing supports the extraction of multiple rows or columns, enabling data filtering, feature selection, or batching.

---

## ⚙️ Operations and Transformations

Two-dimensional tensors support a range of mathematical operations used throughout neural network computations:

- **Element-wise addition** is performed between tensors of the same shape, following the structure of matrix addition.
- **Scalar multiplication** scales each value in the tensor uniformly.
- **Element-wise multiplication** (Hadamard product) is used to apply per-position masking or transformations.
- **Matrix multiplication** follows linear algebra rules and is essential for transforming input features through learned weight matrices in neural network layers.

The ability to perform these operations efficiently is critical for forward passes, gradient computation, and parameter updates during model training.

---

## ✅ Key Takeaways

- Two-dimensional tensors are essential for representing structured data and image grids.
- The row-column layout aligns naturally with both datasets and visual data formats.
- Tensors can be inspected, reshaped, and indexed to control how data flows through a model.
- Core arithmetic operations—including addition, scaling, and matrix multiplication—mirror linear algebra and support deep learning computations.
- The structure and behavior of 2D tensors form the basis for many transformations used in early layers of neural networks.
