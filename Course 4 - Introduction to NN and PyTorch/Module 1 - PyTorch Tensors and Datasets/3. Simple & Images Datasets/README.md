# 🗃️ Module 1 – Section 4: Dataset

This section explores how to construct custom datasets in PyTorch, covering both numerical and image-based data. It explains how to build dataset classes, apply data transformations, and work with TorchVision’s dataset and transform utilities. These practices support scalable, reusable, and consistent data loading pipelines for machine learning workflows.

---

## 📦 Custom Dataset for Numerical Data

A custom dataset is implemented by subclassing PyTorch’s `Dataset` class. This provides a structured interface for storing and retrieving input features and targets using indexing.

### Structure and Behavior

- Feature and target values are stored as tensors (`x`, `y`) during initialization.
- The total number of samples is stored in a `length` attribute.
- Two standard methods are implemented:
  - `__len__`: Returns the number of samples.
  - `__getitem__`: Returns a tuple of tensors `(x, y)` for a given index.

The dataset behaves like a Python iterable. It supports both index access and looping, triggering the `__getitem__` method during each retrieval.

---

## 🔁 Applying Transforms to Dataset Samples

Transformations are implemented using callable classes, which define a `__call__` method. This enables the transformation object to act like a function.

### Transform Class Behavior

- One parameter adds a constant to the input tensor.
- Another parameter multiplies the output tensor by a constant.
- Each time a sample passes through the transform, the modification is applied and the transformed tuple is returned.

### Methods of Application

- **Manual Application**: The transformation object is created separately and applied to individual samples after retrieval.
- **Automatic Application**: The transformation object is passed into the dataset constructor. When `__getitem__` is called, the transformation is applied automatically to every sample.

---

## 🔗 Composing Multiple Transforms

Transform composition enables chaining multiple preprocessing steps using `transforms.Compose`.

- A list of transformation objects is passed to the constructor of `Compose`.
- The composed object applies each transform in sequence.
- The final result is returned as a fully transformed sample.

This composition can also be passed directly into the dataset constructor to automate preprocessing on each sample retrieval.

---

## 🖼️ Custom Dataset for Image Data

In image-based workflows, the dataset is structured similarly but loads samples from disk using metadata from a CSV file.

### Fashion-MNIST Example

- Images: 28×28 grayscale clothing images
- Metadata: CSV file where each row contains:
  - A class label (e.g., "T-shirt", "Dress")
  - The filename of the corresponding image

### Dataset Class Structure

- Loads the CSV into a Pandas DataFrame (`self.data_names`)
- In `__getitem__`:
  - Constructs the full path to the image
  - Opens the image using `Image.open`
  - Retrieves the label from the DataFrame
  - Returns `(image, label)` as a tuple

This method enables memory-efficient loading by retrieving only one image per sample access.

---

## 🧰 TorchVision Transforms

TorchVision provides built-in transforms for preparing images before model input:

- **Cropping**, **scaling**, and **tensor conversion** are supported.
- Transforms can be chained using `Compose`, allowing them to be applied in sequence.
- When the composed transform is passed into the dataset constructor, it is applied automatically during sample retrieval.

Transformed image tensors include an extra dimension for channel/batch compatibility, which aligns with model expectations.

---

## 📚 TorchVision Built-in Datasets

TorchVision also includes standard datasets like **MNIST**, which can be used directly for training and evaluation.

Key parameters include:

- `root`: Directory where data is stored or downloaded
- `train`: Boolean indicating whether to load training or testing data
- `download`: If `True`, dataset will be downloaded if not present
- `transform`: Optional transform pipeline to apply on retrieval

This utility enables fast and reliable access to standardized datasets for benchmarking or development.

---

## ✅ Key Takeaways

- PyTorch supports custom datasets for both structured and image-based data through a clean and extensible interface.
- Transformations can be modularized and applied either manually or automatically during dataset access.
- Compose pipelines enable flexible, sequential data preprocessing.
- Image datasets should load images on-demand using filenames and labels from a metadata file.
- TorchVision provides tools for image transformation and access to widely used benchmark datasets.
- These practices are reusable across project types and scale effectively to large datasets.
