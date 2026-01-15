# 📦 Module 1 – Data Handling

**AI Capstone Project with Deep Learning**

---

## 📌 Overview

Module 1 establishes the **data ingestion and preprocessing foundation** for the AI Capstone Project.  
The work in this module focuses on designing, implementing, and evaluating **scalable data loading and augmentation pipelines** for satellite image classification.

Across three labs, different strategies and frameworks are explored to understand how data loading decisions affect **memory usage, training efficiency, GPU utilization, and system scalability**. The module progressively moves from conceptual comparisons to **production-grade implementations** in both **Keras** and **PyTorch**.

---

## 🧠 Core Concepts and Engineering Decisions

This module treats data loading as a **first-class engineering concern**, equivalent in importance to model architecture and evaluation. The following principles are established and applied:

- Data loading strategy directly impacts scalability and performance.
- Lazy (sequential) loading is required for real-world deep learning systems.
- Augmentation belongs in the data pipeline, not in model logic.
- Framework abstractions differ, but engineering intent remains the same.

---

## 🔹 Lab 1 – Memory-Based vs Generator-Based Data Loading

### What was implemented

- A **memory-based (bulk) loading pipeline**, where:
  - All images are read from disk.
  - Converted to arrays.
  - Stored entirely in RAM before training.
- A **generator-based (sequential/lazy) loading pipeline**, where:
  - Only file paths are stored in memory.
  - Images are loaded, transformed, and released **per batch** during training.

### What was evaluated

- Memory consumption behavior as dataset size increases.
- Startup latency vs repeated access speed.
- Practical scalability limits of bulk loading.
- Suitability of each approach for research vs production use cases.

### Key takeaway

Bulk loading offers fast access but is fundamentally non-scalable.  
Generator-based loading enables training on datasets far larger than available RAM and is the only viable option for serious deep learning workloads.

---

## 🔹 Lab 2 – Data Loading and Augmentation Using Keras

### What was implemented

- Programmatic dataset access by:
  - Building explicit lists of image file paths.
  - Assigning binary labels for agricultural and non-agricultural classes.
- Dataset shuffling to prevent training bias.
- A **custom Python data generator** implementing:
  - Lazy loading
  - Batch construction
  - On-the-fly preprocessing and augmentation
- A **Keras-native pipeline** using:
  - `image_dataset_from_directory`
  - Built-in batching, shuffling, and optimized data handling
- Creation of training and validation datasets with consistent parameters.

### What was analyzed

- Flexibility and transparency of custom generators.
- Performance, simplicity, and scalability of Keras built-in utilities.
- Trade-offs between manual control and framework-optimized pipelines.

### Key takeaway

Keras provides two viable approaches:

- Custom generators for maximum control and transparency.
- Built-in utilities for production-ready, optimized pipelines with minimal boilerplate.

---

## 🔹 Lab 3 – Data Loading and Augmentation Using PyTorch

### What was implemented

- **Custom PyTorch `Dataset` classes** defining:
  - How individual samples are loaded.
  - How labels are assigned.
  - How transformations are applied.
- Modular data augmentation pipelines using:
  - `torchvision.transforms`
- Dataset creation using:
  - `datasets.ImageFolder` for standard directory structures.
- **DataLoader orchestration**, handling:
  - Batching
  - Shuffling
  - Parallel data loading
- Visual inspection of augmented batches and labels to validate pipeline correctness.

### What was analyzed

- Separation of concerns between `Dataset` and `DataLoader`.
- Differences between custom datasets and convention-based utilities.
- PyTorch’s explicit, modular approach to data handling.

### Key takeaway

PyTorch emphasizes transparency and control:

- `Dataset` defines what a sample is.
- `DataLoader` defines how samples are iterated.
  This design enables production-grade, highly customizable data pipelines.

---

## 🔄 Framework Comparison Summary

| Aspect           | Keras                          | PyTorch                  |
| ---------------- | ------------------------------ | ------------------------ |
| Custom pipeline  | Python generator               | Custom `Dataset`         |
| Built-in utility | `image_dataset_from_directory` | `ImageFolder`            |
| Augmentation     | Integrated in pipeline         | `torchvision.transforms` |
| Orchestration    | Framework-managed              | `DataLoader`             |
| Control level    | Higher-level                   | Lower-level, explicit    |

---

## ✅ Module 1 Outcomes

By completing Module 1:

- Scalable data ingestion pipelines were designed and implemented.
- Memory-based and lazy loading strategies were compared in practice.
- Data augmentation was integrated as part of the preprocessing pipeline.
- Equivalent solutions were implemented in **Keras and PyTorch**.
- Clear framework-level trade-offs between convenience and control were established.

This module provides the **engineering foundation** for all subsequent modeling work, ensuring that CNNs and Vision Transformers trained later in the project operate on efficient, robust, and scalable data pipelines.

---
