# 🧠 Module 1 – Tensors and Datasets

This module introduces the foundational concepts for working with tensors, gradients, and datasets in PyTorch. It establishes the groundwork for deep learning by covering data representation with tensors, automatic differentiation for model training, and custom dataset creation for both structured and image-based inputs.

---

## 🔢 Tensors in PyTorch

Tensors are the core data structure in PyTorch, used to represent inputs, outputs, and trainable parameters within a neural network.

### 1D Tensors

- Represent linear data like feature vectors, time series, or rows in a dataset.
- Support indexing, slicing, reshaping, and conversion to/from NumPy and Pandas.

### 2D Tensors

- Represent matrices such as tabular datasets or grayscale images.
- Used for operations like matrix multiplication, broadcasting, and dot product.

Tensors support a wide range of element-wise and structural operations essential to preprocessing, model development, and computation on both CPU and GPU.

---

## 🧠 Automatic Differentiation with PyTorch

Gradient computation in PyTorch is handled via dynamic graph construction.

- Tensors marked with `requires_grad=True` are tracked through computations.
- Backward propagation automatically computes derivatives for scalar or multivariable functions.
- Key tensor attributes for gradient tracking:
  - `grad`: Stores computed gradients
  - `grad_fn`: References the operation used
  - `is_leaf`: Indicates graph position
  - `requires_grad`: Enables tracking

This system is essential for optimization during neural network training.

---

## 🗃️ Building Datasets

PyTorch datasets are built by subclassing the `Dataset` class, allowing flexible access to data.

### Structured (Numerical) Datasets

- Input and target tensors (`x`, `y`) are created in the constructor.
- `__len__` returns sample count; `__getitem__` retrieves indexed data as `(x, y)` pairs.
- Supports transformations via callable classes:
  - Transforms can add/multiply values.
  - Transforms are applied manually or automatically via the dataset constructor.

### Compose and Chained Transforms

- Multiple transforms can be chained using `transforms.Compose`.
- Ensures consistent preprocessing across all dataset samples.

---

## 🖼️ Image Datasets with TorchVision

Custom image datasets are built using metadata from CSV files (image path and label):

- Images are loaded on demand using `Image.open`.
- `__getitem__` returns `(image, label)` as a tuple.

### TorchVision Utilities

- Built-in transforms (e.g., cropping, scaling, tensor conversion) support preprocessing pipelines.
- TorchVision datasets like MNIST and Fashion-MNIST are available for immediate use, supporting training/test splits and automatic downloading.

Transforms and prebuilt datasets ensure compatibility with neural network inputs and accelerate prototyping.

---

## ✅ Key Takeaways

- PyTorch tensors are foundational for data representation and model computation.
- Automatic differentiation tracks operations and enables gradient-based learning.
- Custom datasets support structured and image-based inputs using consistent interfaces.
- Transforms provide modular, scalable preprocessing pipelines.
- TorchVision offers tools and datasets for standardized image handling.

This module provides the essential building blocks for working with neural networks in PyTorch.
