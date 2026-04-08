# 🧠 Module 3 – CNN + Vision Transformers Integration

**AI Capstone Project with Deep Learning**

---

## 📌 Overview

Module 3 builds on the CNN models from Module 2 and introduces a new type of architecture: **Vision Transformers (ViTs)**.

Instead of replacing CNNs completely, the focus here is to **combine both approaches** into a hybrid model:

- CNN → good at detecting **local patterns** (edges, textures, shapes)
- Transformer → good at understanding **relationships across the whole image**

The goal is simple:  
improve classification performance by giving the model both **local understanding + global context**.

Across the three labs, the same idea is implemented twice (Keras and PyTorch), and then compared in a controlled way.

## 🧠 Conceptual Foundation

Up to this point, CNNs were doing all the work.

CNNs process images using filters that look at **small regions at a time**. This works well for things like:

- textures
- edges
- repeated patterns (like crops in satellite images)

But there is a limitation:

> CNNs don’t naturally “see the whole image at once”

They need many layers to build that global understanding, and even then it’s indirect.

### 🔹 What Vision Transformers change

Vision Transformers take a completely different approach:

1. The image is split into **small patches** (like tiles)
2. Each patch is treated as a **token** (similar to words in NLP)
3. The model uses **attention** to understand how patches relate to each other

Instead of scanning locally like CNNs, the model can directly answer questions like:

- “How is this region related to that other region?”
- “Is there a pattern across distant parts of the image?”

### 🔹 Why not just use ViTs alone?

Because ViTs:

- Need **a lot of data**
- Don’t have built-in assumptions about images (no inductive bias like CNNs)

That’s why this module uses a **hybrid approach**:

- CNN → extracts meaningful visual features first
- Transformer → connects those features globally

Think of it as:

> CNN = feature extractor  
> Transformer = relationship model

---

## 🔹 Lab 1 – Vision Transformers using Keras

### What was implemented

This lab builds the hybrid model using **Keras**, focusing on clarity and structure.

Main steps:

- **CNN feature extraction**

  - A pre-trained CNN is used to process the image
  - Instead of using the final output, an **intermediate feature map** is taken
  - This feature map already contains useful visual patterns

- **Convert features into tokens**

  - The feature map (height × width × channels) is reshaped into a sequence
  - Each position becomes a “token”

- **Embedding**

  - A Dense layer transforms tokens into a consistent embedding size

- **Positional encoding**

  - Since order matters, positional information is added
  - Otherwise, the model would treat all tokens as unordered

- **Transformer block**

  - Self-attention → allows tokens to interact
  - Feed-forward layers → process the result
  - Normalization + residual connections → stabilize training

- **Classification**

  - A special token (CLS) is used to summarize the sequence
  - This is passed to a final classification layer

- **Training**
  - Done using Keras `fit()`, keeping the workflow simple

### What was analyzed

- How CNN features behave when passed into a transformer
- How architecture choices affect performance:
  - embedding size
  - number of attention heads
  - number of transformer layers

### Key takeaway

Keras makes it easy to **assemble complex architectures without worrying too much about the training loop**, which is useful to understand the structure before going deeper.

---

## 🔹 Lab 2 – Vision Transformers using PyTorch

### What was implemented

The same idea is implemented again, but now in **PyTorch**, where everything is more explicit.

Main differences:

- **CNN is defined manually**

  - Using `nn.Module`
  - Full control over layers and forward pass

- **Token creation**

  - CNN output is reshaped step-by-step into tokens
  - No abstraction — you see exactly how tensors change

- **Embedding + positional encoding**

  - Implemented manually
  - Includes adding the CLS token

- **Transformer block**

  - Built using PyTorch layers
  - Attention, MLP, normalization, residuals all defined explicitly

- **Data pipeline**

  - Dataset + DataLoader
  - Separate transforms for train/validation

- **Training loop**
  - Forward pass
  - Loss calculation
  - Backpropagation
  - Optimizer step

Everything is handled manually.

### What was analyzed

- How data flows through the model (tensor shapes, transformations)
- Training behavior when everything is explicitly controlled
- How small changes in hyperparameters impact results

### Key takeaway

PyTorch forces you to understand what is happening at each step.  
It’s more work, but also much clearer what the model is actually doing.

---

## 🔹 Lab 3 – Comparative Analysis of CNN–ViT Models (Keras vs PyTorch)

### What was implemented

This lab is not about training — it’s about **evaluation done properly**.

The same trained models (Keras and PyTorch) are compared using:

- The **same dataset**
- The **same preprocessing**
- The **same metrics**

### Evaluation pipeline

- Run inference → get prediction probabilities
- Apply threshold → convert to class labels
- Compute metrics:

  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

- Analyze:
  - Confusion matrix → where the model is making mistakes
  - ROC curve → how performance changes with threshold

### What was analyzed

- Where each model fails (false positives vs false negatives)
- How sensitive results are to threshold choice
- Whether one framework actually performs better

### Key takeaway

Once everything is controlled:

> Keras and PyTorch models behave almost the same

Any difference usually comes from:

- training setup
- hyperparameters
- randomness

Not from the framework itself.

---

## 🔄 Framework Comparison Summary

| Aspect             | Keras CNN–ViT    | PyTorch CNN–ViT      |
| ------------------ | ---------------- | -------------------- |
| Model definition   | High-level API   | `nn.Module`          |
| Training loop      | Managed (`fit`)  | Explicit             |
| Data pipeline      | `tf.data`        | Dataset + DataLoader |
| Transformer design | More abstracted  | Fully explicit       |
| Control level      | Easier to use    | More control         |
| Debugging          | Less transparent | Fully traceable      |

---

## ✅ Module 3 Outcomes

By completing Module 3:

- Vision Transformers were introduced as a way to model **global relationships in images**
- CNN + Transformer hybrid models were built in both frameworks
- The role of CNNs (feature extraction) vs Transformers (context modeling) became clear
- Full training pipelines were implemented in Keras and PyTorch
- Model comparison was done under **controlled conditions**
- It became clear that:
  - frameworks don’t define performance
  - the pipeline and design choices do
