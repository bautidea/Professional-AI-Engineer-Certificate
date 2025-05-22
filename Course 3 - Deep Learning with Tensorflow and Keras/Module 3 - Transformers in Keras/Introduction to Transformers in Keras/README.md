# 🤖 Module 3: Introduction to Transformers in Keras

This section introduces the transformer model, a powerful deep learning architecture widely used for working with sequential data such as text, time series, and audio. Unlike traditional models like RNNs or LSTMs, transformers process sequences more efficiently and handle long-range dependencies better.

---

## 🔷 What Are Transformers?

Transformers are designed to process entire sequences of data at once, using a mechanism called **self-attention** to understand how different parts of the sequence relate to each other.

They are the foundation of modern models like:

- **BERT** (used for language understanding)
- **GPT** (used for language generation)
- **Vision Transformers** (used for image tasks)

---

## 🧱 Core Components of a Transformer

### 🔸 Self-Attention Mechanism

- Each element in a sequence can "attend" to every other element.
- Attention scores are calculated using three vectors: **Query**, **Key**, and **Value**.
- These scores determine how much focus the model gives to each part of the input.

### 🔸 Positional Encoding

- Since transformers process data in parallel, positional encoding is added to the input so the model understands the order of the sequence.

---

## 🔧 Encoder and Decoder Structure

### 🔹 Encoder

- Takes the input sequence.
- Applies self-attention to relate all parts of the input.
- Uses a feed-forward layer to process the attention output.
- Applies residual connections and normalization for stable training.

### 🔹 Decoder

- Takes the target/output sequence.
- Applies self-attention and cross-attention (uses encoder output).
- Processes results through feed-forward layers.
- Also includes residual connections and normalization.

---

## 🛠️ Implementation in Keras

Transformers are built in Keras by defining reusable components:

### 🔹 Multi-Head Attention

- Processes attention across multiple "heads" in parallel.
- Each head focuses on different parts of the sequence.

### 🔹 Transformer Block

- Combines multi-head attention with a feed-forward network.
- Includes normalization and residual paths.

### 🔹 Encoder Layer

- Stacks the transformer block with self-attention and feed-forward layers.
- Uses normalization and residuals for each sub-layer.

---

## 📈 Applications for Sequential Data

Transformers are well suited for:

- **Text**: Understanding or generating language.
- **Time Series**: Forecasting future values based on past sequences.
- **Audio**: Transcribing speech or understanding sound patterns.

They handle these tasks better than older models by allowing full parallel processing and by understanding relationships across long sequences.

---

## ✅ Key Takeaways

- Transformers are used for processing sequences like text and time series.
- They rely on self-attention and positional encoding.
- The model includes an encoder and decoder, both made up of layers with attention and feed-forward networks.
- Transformers are implemented in Keras using modular components like multi-head attention and transformer blocks.
- They are now the base of most state-of-the-art models in AI.