# 🧠 Generative AI and LLMs — Architecture and Data Preparation

## 📦 Module 2: Data Preparation for LLMs

---

## 📌 Overview

This module focuses on how raw text data is transformed into a format that large language models (LLMs) can actually process. Unlike traditional machine learning workflows where data may already be structured, NLP systems require an explicit pipeline that converts human language into numerical representations.

The module builds this pipeline step by step:

- First, text is broken into smaller units (tokenization)
- Then, those units are converted into numerical form (indexing)
- Finally, data is organized and delivered efficiently to the model (data loaders)

Alongside this technical pipeline, the module also emphasizes that the **quality and diversity of data directly affect how well the model performs**, especially in terms of accuracy, bias, and generalization.

---

## 🧩 Conceptual Foundation

LLMs do not understand raw text. They operate on numbers.

This means every sentence must go through a transformation pipeline:

    text → tokens → indices → tensors → model input

Each step in this pipeline changes how the data is represented:

- Text → human-readable
- Tokens → structured pieces of text
- Indices → numerical representation
- Tensors → format usable by deep learning models

If any step is poorly designed, the model will learn incorrect or incomplete patterns.

---

## 🔹 Tokenization

Tokenization is the process of breaking text into smaller units called tokens. These tokens are the first structured representation of text that the model can work with. :contentReference[oaicite:0]{index=0}

For example:

- Text → `"IBM taught me tokenization"`
- Tokens → `["IBM", "taught", "me", "tokenization"]`

The tokenizer is the component responsible for this transformation, and different models use different tokenization strategies.

### Tokenization Methods

Different methods define how text is split and how meaning is preserved.

#### Word-Based Tokenization

Splits text into full words.

- Preserves meaning clearly
- Simple to implement

Limitations:

- Large vocabulary size
- Similar words treated as different tokens (e.g., _unicorn_ vs _unicorns_)

#### Character-Based Tokenization

Splits text into individual characters.

- Small vocabulary
- Handles unseen words easily

Trade-offs:

- Characters carry less meaning
- Sequences become longer
- Higher computational cost

#### Subword-Based Tokenization

Balances word-level meaning and character-level flexibility.

- Keeps common words intact
- Splits rare words into smaller units

This approach:

- Reduces vocabulary size
- Handles unknown words more effectively

Common algorithms include:

- WordPiece → splits based on usefulness and frequency
- Unigram → reduces vocabulary iteratively
- SentencePiece → segments text and assigns token IDs

This combination allows models to represent language efficiently without losing too much semantic information.

### Tokenization → Indexing

After tokenization, tokens must be converted into numbers.

This process involves:

- Building a vocabulary from tokens
- Assigning each token a unique index
- Mapping tokens → indices

Unknown tokens are handled using a default value:

- `<unk>` is used when a word is not in the vocabulary

The result is:

- A tokenized sentence
- A corresponding list of numerical indices

This is the actual input used during training.

### Special Tokens and Padding

Additional tokens are introduced to structure sequences:

- `<bos>` → beginning of sentence
- `<eos>` → end of sentence
- `<pad>` → padding token

Padding ensures all sequences in a batch have the same length:

- Short sequences are extended to match the longest one
- This allows consistent tensor shapes during training

---

## 🔹 Data Loaders

Once text is converted into numerical form, it must be delivered efficiently to the model. This is handled by data loaders.

A data loader automates:

- Batching
- Shuffling
- Data retrieval

It acts as the connection between the dataset and the training loop.

### Purpose of Data Loaders

Instead of processing one sample at a time, data loaders:

- Group samples into batches
- Shuffle data to avoid learning order-based patterns
- Load data on demand (improves memory usage)
- Integrate directly into training pipelines

This is especially important for large NLP datasets.

### Dataset and Data Splitting

Before using a data loader, data is organized into datasets.

Typical splits include:

- Training set → used to learn patterns
- Validation set → used to tune performance
- Test set → used for final evaluation

Each split serves a different purpose in the training process.

### Custom Dataset in PyTorch

In PyTorch, datasets define how data is accessed.

A dataset typically:

- Stores the data (e.g., sentences)
- Returns dataset size
- Retrieves a specific sample by index

This allows the dataset to behave like a list, where elements can be accessed individually.

### DataLoader and Iteration

The DataLoader works as an iterator over the dataset.

It:

- Retrieves batches of data
- Supports iteration (`iter`, `next`)
- Returns a new batch at each step

Key parameters:

- `batch_size` → number of samples per batch
- `shuffle` → randomizes data order

This allows efficient batch-based training instead of single-sample processing.

### Data Transformation in Batches

In NLP workflows, preprocessing is usually applied during batching.

Typical transformations include:

- Tokenization
- Converting tokens to indices
- Padding sequences
- Converting to tensors

This ensures data is correctly formatted before reaching the model.

### Collate Function

The collate function defines how individual samples are combined into a batch.

It is responsible for:

- Tokenizing each sample
- Mapping tokens to indices
- Padding sequences
- Returning a tensor batch

This allows preprocessing to happen dynamically without modifying the dataset itself.

### Data Pipeline Workflow

The complete pipeline follows this structure:

    Dataset → DataLoader → Collate Function → Model Input

Each component has a specific role:

- Dataset → stores and provides access to data
- DataLoader → batches and iterates
- Collate → transforms and formats

---

## 🔹 Data Quality and Diversity

Beyond technical preprocessing, the effectiveness of an LLM depends heavily on the data it is trained on.

### Data Quality

High-quality data ensures reliable model behavior.

Key practices include:

- Noise reduction → removing irrelevant or incorrect data
- Consistency checks → ensuring uniform representation
- Accurate labeling → avoiding misleading signals

Poor data quality introduces noise, which reduces model accuracy.

### Data Diversity

Diverse datasets improve generalization and reduce bias.

This involves:

- Including different demographics and perspectives
- Using multiple data sources (news, social media, literature)
- Incorporating regional and linguistic variation

Without diversity, models may produce biased or narrow outputs.

### Regular Updates

Language evolves over time, so datasets must be updated.

Updates ensure:

- Inclusion of new vocabulary
- Alignment with current cultural norms
- Improved model relevance

Models trained on outdated data may produce inaccurate or biased outputs.

### Ethical Considerations

Data collection must follow ethical practices.

This includes:

- Protecting user privacy (anonymization)
- Ensuring fair representation
- Maintaining transparency in data sources

These practices are essential to build trustworthy AI systems.

---

## 🎯 Outcomes

This module establishes the full data preparation pipeline required before any LLM can be trained or used effectively.
