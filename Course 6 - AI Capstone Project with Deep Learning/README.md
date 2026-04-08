# 🧠 AI Capstone Project with Deep Learning

---

## 📌 Overview

This project brings together all the concepts from the program into a single, end-to-end deep learning workflow.

The objective is to solve a real-world problem:

> Estimate how much of a geographical region corresponds to **agricultural land**, using satellite images.

This is useful in contexts like fertilizer sales forecasting, where understanding land usage directly impacts business decisions.

The project is structured in modules, each one focusing on a different part of the pipeline:

1. **Data loading and preprocessing**
2. **CNN model development**
3. **Vision Transformers and hybrid models**

Instead of focusing only on model performance, the project emphasizes:

- building **scalable pipelines**
- understanding **how models actually work**
- comparing approaches under **controlled conditions**

## 🌍 Problem Context

The task is a **binary image classification problem**:

- Input → satellite image tile
- Output → agricultural / non-agricultural

Even though it looks simple, the challenge is that:

- patterns are **not obvious**
- images can be noisy
- relevant features are often **spatial (textures, shapes, distributions)**

This makes it a good use case for deep learning.

---

## 🧱 Module 1 – Data Handling

This module focuses on how data is loaded and prepared before training.

This is not just a preprocessing step — it directly affects:

- memory usage
- training speed
- scalability

### What was done

- Compared **bulk loading vs lazy loading**

  - Bulk → load everything into memory
  - Lazy → load data per batch during training

- Built data pipelines in:

  - **Keras**
  - **PyTorch**

- Implemented **data augmentation**
  - Random transformations applied during training
  - Helps the model generalize better

### Key idea

> You can’t train a good model if your data pipeline doesn’t scale

Lazy loading + augmentation becomes the standard approach.

---

## 🧠 Module 2 – CNN Model Development

This module introduces **Convolutional Neural Networks (CNNs)** as the baseline model.

CNNs are designed to work with images by learning patterns such as:

- edges
- textures
- shapes

### What was done

- Built a full CNN pipeline in:

  - **Keras**
  - **PyTorch**

- Designed custom CNN architectures

- Trained models and analyzed:

  - loss curves
  - overfitting / underfitting

- Evaluated models using:

  - accuracy
  - precision
  - recall
  - F1-score
  - ROC-AUC

- Compared both frameworks under the same conditions

### Key idea

> CNNs are strong baseline models, but they mainly capture **local patterns**

They don’t naturally understand relationships across the whole image.

---

## 🔁 Module 3 – CNN + Vision Transformers

This module introduces **Vision Transformers (ViTs)** and combines them with CNNs.

### Why this matters

CNNs:

- Look at small regions at a time

Transformers:

- Look at the entire image and understand relationships between distant regions

Combining both gives the model:

- local feature extraction (CNN)
- global context understanding (Transformer)

### What was done

- Built hybrid **CNN + Transformer models** in:

  - Keras
  - PyTorch

- Converted CNN feature maps into **tokens**
- Applied **self-attention** to model relationships between regions

- Evaluated models using the same metrics as Module 2

- Compared both implementations under controlled conditions

### Key idea

> Better performance comes from combining complementary ideas, not replacing them

CNN + Transformer works better than using either one alone (in this context).

## 🔄 Framework Comparison (Keras vs PyTorch)

Throughout the project, both frameworks were used to solve the same problems.

This makes the differences very clear:

| Aspect         | Keras              | PyTorch               |
| -------------- | ------------------ | --------------------- |
| Model building | High-level, faster | More manual, flexible |
| Training       | Managed (`fit`)    | Explicit loop         |
| Data handling  | Built-in utilities | Dataset + DataLoader  |
| Debugging      | Less transparent   | Fully traceable       |

### Key insight

> Performance does not come from the framework  
> It comes from how the model and pipeline are designed

## 📊 Model Evaluation Approach

Across all modules, evaluation was done consistently:

- Same dataset
- Same preprocessing
- Same metrics

Metrics used:

- Accuracy → overall correctness
- Precision → how many predicted positives are correct
- Recall → how many actual positives are detected
- F1-score → balance between precision and recall
- ROC-AUC → performance across thresholds

Additional analysis:

- Confusion matrix → type of errors
- ROC curves → threshold behavior

### Key idea

> If evaluation is not controlled, comparisons are meaningless

---

## ✅ What This Project Demonstrates

- Ability to build **end-to-end deep learning pipelines**
- Understanding of:
  - data engineering for ML
  - model architecture design
  - training and evaluation
- Experience working with:
  - Keras (high-level workflows)
  - PyTorch (low-level control)
- Ability to **compare models correctly**, avoiding common mistakes

---

## 🧭 Final Takeaways

- Data pipelines matter as much as models
- CNNs are strong but limited to local patterns
- Transformers bring global understanding
- Hybrid models combine the best of both
- Framework choice is about workflow, not performance
- Controlled evaluation is critical for real conclusions
