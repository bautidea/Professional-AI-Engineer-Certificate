# 🧠 Module 1 — Fundamentals of Language Understanding

_Foundational Models for NLP & Language Understanding_

## 📘 Overview

This module introduces the first layer of language understanding in NLP systems: converting text into numerical representations, using those representations inside neural networks, and training models to perform classification or next-word prediction.

The module begins with document classification, where raw text is transformed into token indices, embeddings, document vectors, and class scores. It then introduces training, where loss functions and optimization adjust the model parameters so the classifier becomes more reliable. The second part extends language understanding from document-level prediction to word-level prediction with N-Gram models. Instead of predicting a document category, the model receives previous words as context and predicts the most likely next word.

Across the module, the same core idea appears in different forms:

```text
text → numerical representation → neural network → output scores → prediction
```

For document classification, the output scores correspond to categories. For language modeling, the output scores correspond to words in the vocabulary. This progression connects feature representation, neural network architecture, training behavior, and language modeling into one learning path.

---

## 🧩 Topics and Concepts

### 🔹 Text Representation for Neural Networks

Neural networks cannot process raw words directly. Text must first be converted into numerical features so the model can apply matrix operations, learn parameters, and produce predictions.

The simplest representation is **one-hot encoding**. A vocabulary assigns each token a fixed index, and each word is represented as a sparse vector where only one position is active. This gives the model a valid numerical input, but it only identifies the word. It does not represent meaning, similarity, or context.

A **bag-of-words** representation extends one-hot encoding from individual words to full documents. The vectors for the words in a document are summed or averaged into a single document-level vector. This allows the model to detect which words appear in a document, which is useful for simple classification tasks.

The main limitation is that bag of words loses sequence information:

- it can represent word presence or frequency
- it ignores word order
- it does not preserve relationships between neighboring tokens

This makes bag of words useful for document classification, but less appropriate for language modeling, where word order directly changes the predicted next word.

---

### 🔹 Embeddings and Document Representations

Embeddings replace sparse word representations with dense learned vectors. Instead of building a large one-hot vector, the model receives a token index and uses it to retrieve a row from an embedding matrix.

The embedding matrix stores one vector per token:

- each row corresponds to a word in the vocabulary
- each column corresponds to a learned feature dimension
- each vector can be adjusted during training

This makes embeddings more compact than one-hot vectors and allows words that appear in similar contexts to develop more similar representations.

Two PyTorch layers are especially important:

```text
nn.Embedding
    token indices → one vector per token → sequence representation

nn.EmbeddingBag
    token indices + offsets → aggregated vector → one vector per document
```

`nn.Embedding` keeps one row per token, so a document becomes a sequence of word vectors. This structure is useful when later layers need to process token order or word-level relationships.

`nn.EmbeddingBag` aggregates token embeddings into one document-level vector. This is useful for document classification because the output already has one row per document and can be passed directly into a linear classifier. When multiple documents are flattened into a single tensor of token indices, offsets mark where each document begins so the embedding bag can aggregate the correct tokens for each sample.

---

### 🔹 Neural Networks for Document Classification

A document classifier receives a numerical representation of a document and produces one score for each possible category. These raw scores are called **logits**.

The classifier follows a simple flow:

```text
raw text
      ↓
token indices
      ↓
embedding bag
      ↓
document vector
      ↓
fully connected layer
      ↓
class logits
      ↓
predicted category
```

Each output logit corresponds to one category. The values are not probabilities by themselves; they are raw scores indicating how strongly the model associates the document with each class. The predicted class is selected using `argmax`, which returns the index of the highest logit.

This architecture shows how document classification is built from the feature representations introduced earlier. Token indices are converted into document vectors, and those document vectors are transformed into class scores.

---

### 🔹 Learnable Parameters, Hyperparameters, and Training

A neural network transforms data using numerical values that are updated during training. These values are called **learnable parameters** and include weights, biases, and embedding vectors.

Training begins with parameters that do not yet produce reliable predictions. The model generates logits, the loss function measures how wrong those logits are compared with the correct labels, and the optimizer updates the parameters to reduce future loss.

The main training flow is:

```text
prediction → loss → gradients → parameter update
```

**Cross-entropy loss** is used for multi-class classification. It compares the model’s class scores with the correct label and produces a larger penalty when the model assigns high confidence to the wrong class. In PyTorch, `CrossEntropyLoss` accepts raw logits and true labels directly, while internally handling the probability transformation needed for the comparison.

**Optimization** updates the model parameters based on the loss. Gradient descent moves parameters in the direction that reduces prediction error, while the learning rate controls the size of each update. During training, loss should generally decrease and accuracy should increase as the model learns more useful representations.

Hyperparameters are configuration choices set before training begins. They are not learned directly by the optimizer, but they strongly influence training behavior. Examples include:

- embedding dimension
- number of hidden layers
- number of neurons
- learning rate
- batch size
- number of epochs

---

### 🔹 PyTorch and Torchtext Classification Pipeline

The practical classification pipeline connects the conceptual model with PyTorch and Torchtext objects.

Torchtext provides text datasets such as AG News, where each sample contains a document and a category label. Before the text can enter the model, it is processed into token indices:

```text
raw text → tokenizer → vocabulary lookup → token indices
```

A custom batching function prepares examples for `nn.EmbeddingBag`. It collects labels, concatenates token indices, and creates offsets so the model can recover document boundaries inside the flattened token tensor.

The main implementation components are:

- `tokenizer`: splits raw text into tokens
- `vocab`: maps tokens to integer indices
- `collate_batch`: prepares labels, token indices, and offsets
- `DataLoader`: groups samples into batches
- `nn.EmbeddingBag`: turns token indices into document vectors
- `nn.Linear`: maps document vectors to class logits
- `CrossEntropyLoss`: compares logits with true labels
- optimizer: updates learnable parameters

The dataset is divided into training, validation, and test subsets. Training data updates the model parameters, validation data monitors whether the model is improving beyond the training examples, and test data evaluates final performance after training. When validation accuracy improves, the current parameters can be saved as the best observed model state.

---

### 🔹 Language Modeling with N-Grams

Document classification predicts a category for an entire document. Language modeling predicts the next word from previous words. This changes the role of the model output: instead of producing one score per document class, the model produces one score per word in the vocabulary.

An N-Gram model uses a fixed context window:

- **Bi-Gram**: uses one previous word
- **Tri-Gram**: uses two previous words
- **N-Gram**: uses `N-1` previous words

A Bi-Gram model can estimate what usually follows a word such as `like`, but it cannot distinguish between contexts such as `I like` and `surgeons like` because it only looks one word back. A Tri-Gram model uses two previous words, so it can make different predictions depending on the larger context.

Larger context windows provide more information, but they also create more possible word combinations. As the number of combinations grows, direct probability tables become harder to estimate because many contexts may appear rarely or not at all in the training data.

---

### 🔹 Neural N-Gram Models

A neural N-Gram model solves the same next-word prediction problem without storing explicit probability tables for every possible context. The model learns a function that maps context words to output scores over the vocabulary.

The neural N-Gram pipeline is:

```text
context word indices
      ↓
embedding lookup
      ↓
embedding concatenation
      ↓
hidden layer
      ↓
output scores over vocabulary
      ↓
predicted next word
```

The context words are first converted into token indices. Each index retrieves an embedding vector. These embeddings are concatenated into one context vector so the model preserves word position inside the context window.

This is important because bag of words would lose order. For language modeling, `I like` and `like I` should not produce the same representation because they do not imply the same next word.

The context vector size depends on:

```text
context size × embedding dimension
```

For example, with context size `2` and embedding dimension `3`, the context vector has `6` values. The next layer must expect this size as its input dimension.

A neural N-Gram model is essentially a classifier where each class is a vocabulary word. If the vocabulary contains six words, the output layer has six neurons. The neuron with the highest score corresponds to the predicted next word.

---

### 🔹 Sliding Windows and N-Gram Training

To train a neural N-Gram model, text sequences are converted into supervised examples using a sliding window.

For each position in the sequence:

```text
previous words → target word
```

The previous words become the context, and the current word becomes the target the model should predict. As the window moves forward, one sentence can produce many `(context, target)` pairs.

This converts language modeling into a supervised learning task. The model receives context token indices, produces scores over the vocabulary, and the loss function compares those scores with the correct target word.

In this workflow, loss is the main performance signal. Accuracy can be less informative because the model is learning to assign better probability-like scores across many possible words, not only to choose from a small number of document categories.

After prediction, the model outputs a token index. A mapping such as `vocab.get_itos()` converts that index back into a readable word:

```text
predicted index → vocabulary lookup → predicted word
```

Repeating this process allows the model to generate longer sequences one word at a time.

---

## 🛠️ Practical Implementation

Module 1 uses PyTorch and Torchtext to connect language understanding concepts with implementation patterns.

The document classification workflow demonstrates how raw text becomes a model-ready input through tokenization, vocabulary lookup, batching, offsets, embedding bags, and linear classification. The training workflow then shows how cross-entropy loss and gradient-based optimization update model parameters so predictions become more accurate.

The N-Gram workflow uses PyTorch to reframe next-word prediction as a classification task over vocabulary words. Context words are converted into embeddings, concatenated into a context vector, passed through a hidden layer, and mapped to output scores. Sliding windows create the context-target examples needed for training.

Across both workflows, PyTorch objects are used to represent specific roles in the system:

- `DataLoader` organizes samples into batches.
- `collate_batch` converts raw samples into tensors with the structure expected by the model.
- `nn.Embedding` maps token indices to dense word vectors.
- `nn.EmbeddingBag` aggregates token embeddings into document-level vectors.
- `nn.Linear` maps learned representations to output scores.
- `CrossEntropyLoss` measures the difference between model outputs and true labels.
- Optimizers update weights, biases, and embeddings during training.

---

## 🔗 How the Concepts Connect

The module moves from document-level language understanding to word-level language modeling.

In the first part, text is represented as a document vector so a classifier can predict a category. This requires understanding how tokens become indices, how embeddings represent words, how embedding bags produce document-level vectors, and how logits become class predictions.

In the second part, the focus shifts from classifying documents to predicting words. The same neural network idea still applies, but the input and output change. The input becomes a fixed-size context window, and the output becomes a score for every vocabulary word.

The shared pattern is:

```text
text input → numerical representation → learned transformation → output scores → prediction
```

By the end of the module, the learner can see how NLP systems use the same basic neural network mechanics for different language tasks: classifying documents and predicting the next word.

---

## ✅ Takeaways

- **Numerical text representation** is required because neural networks operate on tensors rather than raw words.
- **One-hot encoding** represents token identity with sparse vocabulary-sized vectors.
- **Bag of words** creates document-level representations by aggregating word vectors, but it loses word order.
- **Embeddings** map token indices to dense learned vectors that can improve during training.
- **Embedding bags** aggregate token embeddings into one vector per document, making them useful for document classification.
- **Offsets** preserve document boundaries when multiple documents are stored in a flattened token tensor.
- **Document classifiers** transform document vectors into logits, where each logit corresponds to one category.
- **Cross-entropy loss** compares class scores with true labels and gives the optimizer a training signal.
- **Gradient-based optimization** updates weights, biases, and embeddings to reduce prediction error.
- **N-Gram models** predict the next word from a fixed window of previous words.
- **Neural N-Gram models** treat vocabulary words as output classes and learn next-word prediction from context vectors.
- **Concatenated embeddings** preserve word position inside an N-Gram context window.
- **Sliding windows** convert text sequences into supervised context-target pairs.
- **Module 1** connects text representation, classification, training, and language modeling into a foundation for later NLP architectures.
