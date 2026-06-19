# 🧠 Module 1 — Fundamentals of Language Understanding

## 📝 Section 1 - Language Understanding with Neural Networks

_Foundational Models for NLP & Language Understanding_

## 📘 Overview

This section introduces the first complete NLP classification pipeline: converting raw text into numerical representations, passing those representations through a neural network, producing category scores, and training the model so its predictions become meaningful.

Neural networks cannot process words directly. Text must first be transformed into numerical features that preserve enough information for the model to learn useful patterns. The section begins with basic representations such as one-hot encoding and bag of words, then moves into learned representations using embeddings and embedding bags. These feature representations become the input to a document classification model built with PyTorch and Torchtext.

The learning path follows the full classification workflow:

```text
raw text → tokens → token indices → document representation → neural network → logits → predicted class
```

Once the prediction pipeline is defined, training explains how the model improves. Cross-entropy loss measures how wrong the predictions are, and optimization updates the learnable parameters so the model gradually assigns higher scores to the correct categories.

---

## 🧩 Main Concepts

### 🔹 Text as Numerical Features

Text must be converted into numerical form before it can enter a neural network. A vocabulary assigns each token a fixed index, and that index can be represented in different ways depending on how much information the model needs.

One-hot encoding represents each word as a sparse vector whose length equals the vocabulary size. Only one position is active, and that position identifies the word. This gives the model a valid numerical input, but it only represents identity. It does not capture similarity, context, or relationships between words.

Bag of words extends this idea from individual words to full documents. Instead of keeping each token separately, the one-hot vectors for all words in a document are summed or averaged into one document-level vector. This makes the input suitable for simple classification because the model can detect which words appear in the document.

The trade-off is that bag of words loses sequence information:

- it can represent word presence or frequency
- it ignores the order of words
- it does not preserve context between neighboring tokens

This makes bag of words useful for simple document classification, but limited for tasks where word order changes the meaning.

---

### 🔹 Embeddings and Embedding Bags

An embedding layer replaces sparse one-hot vectors with smaller learned vectors. Instead of building a large vocabulary-sized vector, the model receives a token index and uses it to retrieve a row from an embedding matrix.

The embedding matrix stores one vector per token:

- each row corresponds to a word in the vocabulary
- each column corresponds to a learned feature dimension
- each vector is updated during training

This changes the representation from sparse and fixed to dense and learnable. Words that appear in similar contexts can develop more similar representations because their embedding vectors are adjusted as the model learns.

`nn.Embedding` and `nn.EmbeddingBag` both use embedding matrices, but they return different structures.

`nn.Embedding` returns one vector per token. A document with multiple words becomes a sequence of embedding vectors, where each row represents one word. This is useful when later layers need to process token order or word-level relationships.

`nn.EmbeddingBag` aggregates multiple token embeddings into one document-level vector. This makes it especially useful for document classification because the output already has one row per document and can be passed directly into a linear classifier.

The practical distinction is:

```text
nn.Embedding
    token indices → one embedding per token → sequence representation

nn.EmbeddingBag
    token indices + offsets → aggregated embedding → one vector per document
```

Offsets are needed when multiple documents are flattened into one tensor of token indices. They mark where each document begins, allowing `nn.EmbeddingBag` to aggregate the correct tokens for each document independently.

---

### 🔹 Neural Networks for Document Classification

A document classifier receives a numerical representation of a document and produces one score for each possible category. These raw scores are called logits.

For a news classification task, the output layer may produce one logit for each category, such as business, sports, world, or science and technology. The values are not probabilities by themselves. They are scores that indicate how strongly the model associates the document with each class.

The predicted category is selected with `argmax`, which returns the index of the largest logit:

```text
document representation → logits → argmax → predicted class
```

The architecture used in this section follows a simple classification pattern:

```text
token indices
      ↓
embedding bag
      ↓
document vector
      ↓
fully connected layer
      ↓
class logits
```

The embedding bag creates one dense vector per document. The fully connected layer maps that document vector into class scores. This keeps the architecture simple while still allowing the model to learn which document patterns are associated with each class.

---

### 🔹 Learnable Parameters and Hyperparameters

A neural network transforms data using numerical values that are adjusted during training. These values are called learnable parameters.

Learnable parameters include:

- weights connecting neurons
- bias values
- embedding vectors

During training, these values are updated so the model produces better predictions. They are often represented collectively as `Θ` because modern networks can contain a very large number of parameters.

Hyperparameters are different. They are configuration choices set before training begins and are not directly learned by the optimizer. Examples include:

- number of hidden layers
- number of neurons in a layer
- embedding dimension
- learning rate
- batch size
- number of epochs

These choices affect how much the model can learn, how stable training is, and how efficiently the network processes data.

---

### 🔹 Cross-Entropy Loss

A classifier must measure how far its predictions are from the correct labels. Cross-entropy loss provides this measurement for multi-class classification.

The model first produces logits. These logits can be interpreted through softmax, which converts raw scores into a probability distribution over classes. Softmax makes all values positive and normalizes them so they sum to one.

Cross-entropy compares the model’s predicted distribution with the true class label. When the model assigns high confidence to the correct class, the loss is low. When the model assigns high confidence to the wrong class, the loss becomes large.

This behavior is useful because the loss gives the optimizer a clear signal:

```text
wrong prediction → high loss → larger correction needed
better prediction → lower loss → smaller correction needed
```

In PyTorch, `CrossEntropyLoss` accepts raw logits and true labels directly. PyTorch handles the softmax and logarithmic comparison internally, so the model does not need to manually convert logits into probabilities before computing the loss.

---

### 🔹 Optimization and Gradient Descent

Optimization is the process that updates the model parameters to reduce the loss.

After each batch, the model produces predictions and the loss function measures the error. Backpropagation computes how each parameter contributed to that error. The optimizer then updates the parameters in the direction that should reduce future loss.

Gradient descent follows this logic:

```text
prediction → loss → gradients → parameter update
```

The learning rate controls the size of each update. A larger learning rate changes parameters more aggressively, while a smaller learning rate makes training slower but more stable.

Training usually improves when the loss decreases and accuracy increases over time. This means the model is assigning higher scores to the correct classes and learning more useful internal representations.

---

## 🛠️ Practical Implementation

This section uses PyTorch and Torchtext to connect the conceptual pipeline with implementation objects.

Torchtext provides the AG News dataset, where each sample contains a text document and a category label. Before the model can process the text, the data must move through a preprocessing pipeline:

```text
raw text → tokenizer → vocabulary lookup → token indices
```

A custom batching function prepares the data for `nn.EmbeddingBag`. It collects labels, concatenates token indices, and creates offsets so the model knows where each document begins inside the flattened tensor.

The main implementation components are:

- `tokenizer`: splits raw text into tokens
- `vocab`: maps tokens to integer indices
- `collate_batch`: converts samples into labels, token indices, and offsets
- `DataLoader`: groups samples into batches
- `nn.EmbeddingBag`: converts token indices into one document vector per sample
- `nn.Linear`: maps document vectors to class logits
- `CrossEntropyLoss`: compares logits with the correct labels
- optimizer: updates learnable parameters during training

The model is trained over multiple epochs. Each epoch processes the training data in batches. For each batch, the model generates logits, computes loss, clears old gradients, computes new gradients with `backward()`, and updates parameters with the optimizer.

The dataset is split into training, validation, and test subsets. Training data updates the model parameters, validation data monitors whether the model is improving on unseen examples, and test data evaluates final performance after training. When validation accuracy improves, the current model parameters can be saved as the best observed configuration.

---

## ✅ Takeaways

- **Text feature representation** converts raw language into numerical inputs that neural networks can process.
- **One-hot encoding** identifies words with sparse vocabulary-sized vectors, but it does not capture meaning or similarity.
- **Bag of words** aggregates word vectors into one document representation, making it useful for classification while losing word order.
- **Embeddings** replace sparse vectors with dense learned representations that can improve as the model trains.
- **Embedding bags** aggregate token embeddings into one document-level vector, which can be passed directly into a linear classifier.
- **Offsets** preserve document boundaries when multiple documents are stored in one flattened token tensor.
- **Document classifiers** produce logits, where each logit is a raw score for one possible category.
- **Argmax** selects the predicted class by choosing the category with the highest logit.
- **Cross-entropy loss** measures how strongly the model’s class scores disagree with the correct label.
- **Gradient-based optimization** updates weights, biases, and embeddings so the classifier gradually reduces prediction error.
- **Validation performance** helps identify whether the model is improving beyond the training data and determines which parameters should be saved.
