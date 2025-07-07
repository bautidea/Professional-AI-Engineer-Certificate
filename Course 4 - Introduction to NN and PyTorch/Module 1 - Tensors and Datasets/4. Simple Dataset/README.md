# 🗃️ Module 1 – Section 4: Simple Dataset

This section introduces the process of building a custom dataset class in PyTorch. It explains how to structure feature and target data, retrieve individual samples, and apply transformations using callable objects. The content also demonstrates how to chain multiple transforms efficiently during data loading.

---

## 🧩 Conceptual Overview

A dataset in PyTorch is represented as a class that inherits from the built-in `Dataset` abstraction. This design enables structured access to data using indexing and allows integration with data loaders and training pipelines.

The custom dataset class defines how input data (features) and target values are stored and retrieved. Data is organized as tensors and made accessible through standardized methods for length and sample access.

---

## 📦 Dataset Structure and Behavior

- Features (`x`) and targets (`y`) are stored as tensors inside the dataset object.
- Each dataset contains a defined number of samples (e.g., 100), tracked using a dedicated attribute.
- Two core methods are implemented:
  - `__len__`: Returns the total number of samples.
  - `__getitem__`: Retrieves a sample at a specific index as a tuple of `(x, y)`.

The dataset class behaves like a Python iterable, supporting indexing with brackets and use within loops for iteration over samples.

---

## 🔁 Applying Transforms to Dataset Samples

Transformations can be applied to dataset samples using callable classes that encapsulate operations such as scaling or normalization.

- A transformation class defines a `__call__` method, allowing the object to behave like a function.
- These transforms operate on individual samples, modifying the feature or target tensors as needed.
- Parameters such as additive or multiplicative constants are specified when the transformation object is instantiated.

---

## ⚙️ Integration of Transforms

Two primary methods exist for applying transformations:

### Manual Application

- A transformation object is created separately.
- It is applied manually to individual samples after retrieval from the dataset.
- This method is suitable for inspecting the effect of a transform on specific data points.

### Automatic Application via Constructor

- The transformation object is passed directly into the dataset constructor.
- During each call to retrieve a sample, the transformation is applied automatically.
- This ensures consistency and avoids redundant code during data loading.

---

## 🔗 Composing Multiple Transforms

PyTorch supports sequential transformation through the `Compose` utility:

- A list of transformation objects is passed to the `Compose` constructor.
- When a sample is retrieved:
  - The first transformation is applied to the input.
  - The result is passed to the next transformation.
  - The final output is returned as a modified tuple of tensors.

Composed transformations can also be passed into the dataset constructor for automatic, chained application.

---

## ✅ Key Takeaways

- Custom dataset classes allow for structured and modular data management in PyTorch.
- Features and labels are stored in tensors and accessed through standard methods.
- Transformations can be encapsulated in callable objects and applied during sample retrieval.
- Passing transforms into the dataset constructor enables automatic preprocessing during training.
- Multiple transformations can be chained using composition for clean and scalable data pipelines.
