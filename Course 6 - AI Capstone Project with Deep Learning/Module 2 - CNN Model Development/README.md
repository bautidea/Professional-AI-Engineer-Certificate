# 🧠 Module 2 – Convolutional Neural Network (CNN) Model Development

**AI Capstone Project with Deep Learning**

---

## 📌 Overview

Module 2 focuses on **designing, training, evaluating, and comparing convolutional neural network (CNN) models** for a real-world geospatial image classification problem.  
The objective is to classify satellite image tiles as **agricultural** or **non-agricultural**, supporting fertilizer sales forecasting by estimating farmland coverage in a new territory.

Across three labs, this module establishes **CNNs as the baseline modeling approach** for the capstone and demonstrates how equivalent solutions are implemented and evaluated using **Keras** and **PyTorch**, followed by a **rigorous, framework-agnostic comparison**.

---

## 🌍 Problem Context and Modeling Goal

The work in this module is framed around a business-driven question:

> _What proportion of a new geographical region corresponds to agricultural land?_

To answer this, CNNs are trained on labeled satellite imagery to automatically identify farmland based on subtle spatial and textural patterns such as crop canopies, furrow structures, and irrigation layouts.

CNNs are chosen because of their ability to:

- Learn hierarchical visual features
- Generalize across large spatial areas
- Replace brittle rule-based image analysis

---

## 🔹 Lab 1 – Keras-Based Agricultural Land Classifier

### What was implemented

This lab establishes the **Keras CNN baseline** by building a complete, end-to-end deep learning pipeline using TensorFlow/Keras.

Key components include:

- **Dataset discovery and path management**

  - Recursive traversal of the dataset using `os.walk`
  - Explicit control over image paths and labels

- **Training and validation data generators**

  - Lazy loading using `ImageDataGenerator`
  - Augmentation applied only to the training data
  - Clean, non-augmented validation data for unbiased evaluation

- **Custom CNN architecture**

  - Convolutional and pooling layers for feature extraction
  - Dense layers for binary classification
  - Architecture tailored to small satellite image tiles

- **Model training with managed workflow**

  - Training handled via `model.fit`
  - Integration of callbacks for training control

- **Model checkpointing**

  - Automatic saving of the best-performing model based on validation metrics

- **Training diagnostics**
  - Visualization of training and validation loss and accuracy
  - Identification of overfitting, underfitting, and convergence behavior

### Key takeaway

Keras enables **rapid prototyping and clean experimentation**, allowing the entire CNN training lifecycle to be expressed concisely while maintaining strong performance and reproducibility.

---

## 🔹 Lab 2 – PyTorch-Based Agricultural Land Classifier

### What was implemented

This lab provides a **PyTorch-based CNN implementation** of the same classification task, emphasizing explicit control and transparency.

Key components include:

- **Explicit data augmentation pipelines**

  - Separate training and validation transformations using `torchvision.transforms.Compose`
  - Clear distinction between augmented and non-augmented data flows

- **Dataset and DataLoader construction**

  - Use of directory-structured datasets
  - Batched, shuffled, and parallelized data loading with `DataLoader`

- **CNN architecture defined via `nn.Module`**

  - Manual definition of layers and forward pass
  - Full transparency over tensor flow and computations

- **Custom training loop**

  - Explicit forward pass, loss computation, backpropagation, and optimizer steps
  - Manual GPU handling and gradient management

- **Training diagnostics**

  - Visualization of training metrics to inspect convergence behavior

- **Prediction generation**
  - Explicit inference logic for downstream evaluation and metric computation

### Key takeaway

PyTorch offers **fine-grained control over every stage of training**, making it ideal for debugging, customization, and research-oriented workflows, at the cost of increased verbosity.

---

## 🔹 Lab 3 – Comparative Analysis of Keras and PyTorch Models

### What was implemented

This lab performs a **rigorous, fair comparison** of the pre-trained Keras and PyTorch CNN models.

The focus is on **evaluation discipline**, not training.

Key components include:

- **Standardized evaluation protocol**

  - Same dataset, preprocessing, and labels
  - Identical metrics across frameworks
  - Controlled experimental conditions

- **Unified performance metrics**

  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion matrix

- **Prediction thresholding**

  - Conversion of predicted probabilities into class labels
  - Discussion of how thresholds affect business outcomes

- **Confusion matrix interpretation**

  - Identification of systematic errors
  - Insight into false positives vs false negatives

- **ROC curve visualization**
  - Comparison of class separability across thresholds
  - Threshold-independent performance analysis

### Key takeaway

When evaluation conditions are controlled, **Keras and PyTorch CNNs achieve comparable predictive performance**.  
Observed differences are typically due to **implementation choices and workflows**, not inherent framework superiority.

---

## 🔄 Framework Comparison Summary

| Aspect           | Keras CNN              | PyTorch CNN              |
| ---------------- | ---------------------- | ------------------------ |
| Model definition | Sequential API         | `nn.Module`              |
| Training loop    | Managed (`model.fit`)  | Explicit loop            |
| Data loading     | `tf.data` / generators | Dataset + DataLoader     |
| Augmentation     | High-level utilities   | `torchvision.transforms` |
| Control level    | High-level             | Low-level, explicit      |
| Debuggability    | Abstracted             | Fully transparent        |

---

## ✅ Module 2 Outcomes

By completing Module 2:

- CNNs were established as a **strong baseline** for geospatial image classification
- Equivalent CNN pipelines were implemented in **Keras and PyTorch**
- Training behavior and convergence were analyzed visually and quantitatively
- Model evaluation was grounded in **business-relevant metrics**
- A **fair, reproducible framework comparison methodology** was established

Module 2 provides the **benchmark models and evaluation framework** that will be used to assess more advanced architectures, such as Vision Transformers, in subsequent modules.

---
