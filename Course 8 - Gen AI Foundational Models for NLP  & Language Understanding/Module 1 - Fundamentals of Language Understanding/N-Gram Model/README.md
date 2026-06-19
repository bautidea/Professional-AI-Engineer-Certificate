# 🧠 Module 1 — Fundamentals of Language Understanding:

## 📝 Section 2: N-Gram Model

_Foundational Models for NLP & Language Understanding_

## 📘 Overview

This section introduces language modeling through N-Grams and then shows how the same prediction problem can be implemented with a neural network in PyTorch.

The central problem is next-word prediction. Instead of classifying an entire document into a category, the model receives a small context of previous words and predicts which word is most likely to appear next. This makes language modeling different from document categorization: the output is no longer a document class such as sports or business, but a word from the vocabulary.

The section first explains how Bi-Gram and Tri-Gram models estimate next-word probabilities from observed word sequences. It then extends the idea to larger N-Gram models, where more previous words can be used as context. As the context size grows, direct probability tables become harder to manage, so neural networks are introduced as a way to approximate next-word probabilities from learned representations.

The learning path can be summarized as:

```text
previous words → context representation → neural network → scores over vocabulary → predicted next word
```

---

## 🧩 Main Concepts

### 🔹 Language Modeling with Context

Language modeling is based on the idea that the next word in a sentence depends on the words that came before it. A phrase such as `I like ...` suggests a different continuation than `I hate ...` because the previous words change the expected meaning.

An N-Gram model uses this context to estimate which word is most likely to appear next. The model does not try to understand an entire document at once. Instead, it focuses on a local prediction problem:

```text
given previous words → predict the next word
```

This is the foundation for both traditional N-Gram models and neural N-Gram models.

---

### 🔹 Bi-Gram and Tri-Gram Models

A Bi-Gram model predicts the next word using only the immediately previous word. Its context size is one. If the model sees the previous word `like`, it estimates which words most often follow `like` in the observed data.

This works for simple patterns, but it can miss important context. For example, the phrase `I like ...` and `surgeons like ...` both end with the same immediate previous word, but they may point to different likely completions. A Bi-Gram model cannot distinguish between them because it only looks one word back.

A Tri-Gram model improves on this by using two previous words as context. Instead of asking only what follows `like`, it asks what follows a two-word context such as `I like` or `surgeons like`. This gives the model more information and allows it to make different predictions depending on the larger context.

The idea generalizes naturally:

| Model    | Context Size       |
| -------- | ------------------ |
| Bi-Gram  | 1 previous word    |
| Tri-Gram | 2 previous words   |
| 4-Gram   | 3 previous words   |
| N-Gram   | N-1 previous words |

Larger context windows usually provide more information, but they also create more possible word combinations. As the number of combinations grows, many contexts may appear rarely or not at all in the training data. This makes direct probability estimation difficult.

---

### 🔹 From Probability Tables to Neural Networks

Traditional N-Gram models estimate probabilities by counting how often word combinations appear. This works when the vocabulary and context size are small, but it becomes less practical as the number of possible contexts increases.

A neural network provides another way to solve the same problem. Instead of storing an explicit probability for every possible context, the network learns a function that maps context words to output scores over the vocabulary.

The model still follows the same basic objective:

```text
context words → predicted next word
```

The difference is how the prediction is produced. The neural network receives a numerical context representation, transforms it through learned layers, and outputs one score for each word in the vocabulary. The word with the highest score becomes the predicted next word.

---

### 🔹 Context Vectors and Word Order

A context must be converted into numerical form before it can enter a neural network. One option is to represent each context word as a one-hot vector, but a bag-of-words representation is not appropriate for language modeling because it loses word order.

For example, `I like` and `like I` would produce the same bag-of-words representation, even though they do not represent the same context. In language modeling, position matters because the order of words changes what the next word is likely to be.

To preserve order, the word representations are concatenated instead of aggregated. For a context size of two, the first word representation and the second word representation are joined into a larger context vector. This preserves both:

- which words appear in the context
- where each word appears inside the context window

When embeddings are used, the same idea applies. The model retrieves one embedding vector per context word and concatenates those embeddings into a single context vector.

The context vector size depends on:

```text
context size × embedding dimension
```

For example, if the context size is `2` and each embedding has dimension `3`, the context vector has `6` values.

---

### 🔹 Neural N-Gram Architecture

A neural N-Gram language model can be understood as a classification model where the classes are vocabulary words.

In document classification, the output layer produces one score per category. In neural language modeling, the output layer produces one score per possible next word. If the vocabulary contains six words, the output layer has six neurons.

The architecture follows this flow:

```text
context token indices
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

The embedding layer converts context token indices into dense vectors. These vectors are reshaped or concatenated into a single context vector. A hidden layer then transforms that context representation before the output layer produces scores for every word in the vocabulary.

This makes the model more flexible than a direct count-based table because the network can learn internal patterns between contexts and target words.

---

## 🛠️ Practical Implementation

The PyTorch implementation focuses on turning a sequence of text into supervised training examples.

A sliding window moves across the sequence. At each position, the previous words become the context and the current word becomes the target. For a context size of two, the model uses the two previous words to predict the next word.

The sliding window transforms one sequence into many training pairs:

```text
context words → target word
```

This is the key step that turns language modeling into a supervised learning problem.

The implementation uses a text processing pipeline to convert words into token indices. Instead of relying on a larger dataset object, a simple list can be used to focus on the mechanics of language modeling. Once the text is represented as indices, the model can generate context-target pairs, pass context indices through the embedding layer, and train against the target word index.

Padding is used when inputs need consistent shapes inside a batch. This allows the model to process multiple examples together without shape mismatches.

During training, loss is the main performance signal. Accuracy is less central because the model is learning a probability distribution over possible next words. A lower loss indicates that the model is assigning better scores to the correct target words.

After prediction, the model produces numerical indices rather than readable words. The `vocab.get_itos()` mapping converts predicted indices back into tokens. This mapping works like a decoder:

```text
predicted index → vocabulary lookup → predicted word
```

The same prediction process can be repeated to generate a longer sequence one word at a time.

---

## ✅ Takeaways

- **Language modeling** predicts the next word from previous words used as context.
- **Bi-Gram models** use one previous word as context, which makes them simple but limited.
- **Tri-Gram models** use two previous words, allowing the prediction to depend on a larger context.
- **N-Gram models** generalize this idea by allowing arbitrary context sizes.
- **Larger context windows** can improve prediction but make probability tables harder to estimate directly.
- **Neural N-Gram models** replace explicit probability tables with learned functions that map context representations to vocabulary scores.
- **Bag of words** is not suitable for language modeling because it loses word order.
- **Concatenated context representations** preserve the position of each context word.
- **Embedding layers** convert context token indices into dense vectors before they are concatenated.
- **Neural N-Gram output layers** produce one score for each word in the vocabulary.
- **Sliding windows** convert text sequences into supervised `(context, target)` training pairs.
- **Loss** is the main training signal because the model is learning to assign better scores to the correct next word.
- **Index-to-token mappings** convert numerical predictions back into readable words.
