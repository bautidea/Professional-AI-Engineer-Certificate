# 🧠 Generative AI and LLMs — Architecture and Data Preparation

---

## 📌 Overview

This course builds the foundation for working with generative AI systems and large language models (LLMs), focusing on two core aspects:

- How generative models are structured and generate content
- How raw text data is transformed into a format these models can process

The course follows a clear progression. It starts by explaining how generative AI models work internally, then moves into how text must be prepared before it can be used by those models.

At a system level, the course connects two pipelines:

- **Model side** → how architectures learn and generate outputs
- **Data side** → how text is transformed into numerical inputs

Understanding both sides is necessary to build real NLP applications.

---

## 🧩 Conceptual Foundation

Generative AI models do not operate on raw human-readable text. They rely on learned patterns and structured numerical representations.

The full pipeline can be understood as:

    raw text → structured tokens → numerical representation → model → generated output

This introduces two critical responsibilities:

- The **model** must learn patterns and relationships from data
- The **data pipeline** must convert human language into a format the model can understand

If either side is poorly designed, the system will fail to produce meaningful results.

---

# 📦 Module 1: Generative AI Architecture

## 📌 What is happening inside generative AI models

Generative AI models learn patterns from existing data and use those patterns to generate new content. Instead of predicting labels, these systems produce outputs such as text, images, or audio.

This means the model is not just answering questions like:

- “What class is this?”

It is answering:

- “What should come next?”
- “What output matches this input?”

## 🔹 Generative AI in practice

These models can generate:

- Text → conversations, summaries, translations
- Images → from text prompts or transformations
- Audio → natural-sounding speech
- Other content → 3D objects, music

Applications include:

- Content generation
- Chatbots and assistants
- Language translation
- Data analysis
- Synthetic data creation

These capabilities are applied across industries such as healthcare, finance, gaming, and IT.

## 🔹 Core Architectures

Different architectures define how generative systems learn and generate outputs.

### Recurrent Neural Networks (RNNs)

RNNs process sequential data using loops that allow past information to influence current outputs.

- Capture time or sequence dependencies
- Used in language modeling, translation, speech tasks

### Transformers

Transformers process sequences through layers using self-attention.

- Identify important parts of input
- Process sequences in parallel (not step-by-step)
- Scale efficiently for large datasets

Fine-tuning typically:

- Keeps most layers fixed
- Adjusts output layers for specific tasks

Example:

- GPT for text generation

### Generative Adversarial Networks (GANs)

GANs train two models together:

- Generator → creates synthetic data
- Discriminator → evaluates authenticity

Both improve through competition, making outputs increasingly realistic.

### Variational Autoencoders (VAEs)

VAEs compress data into a latent representation and reconstruct it.

- Learn underlying patterns
- Represent data probabilistically
- Generate variations of input

### Diffusion Models

Diffusion models learn to generate data by reversing noise.

- Training → data is progressively noised
- Model → learns to remove noise step by step

Used for high-quality image generation and reconstruction.

## 🔹 Generative AI for NLP

Generative AI enables systems to understand and generate human language in a coherent way.

These systems:

- Maintain context across sequences
- Interpret meaning beyond individual words
- Generate responses aligned with user intent

### Evolution of NLP systems

- Rule-based → fixed rules, limited flexibility
- Machine learning → statistical patterns
- Deep learning → neural networks
- Transformers → context-aware sequence modeling

## 🔹 Large Language Models (LLMs)

LLMs are transformer-based models trained on large-scale text data.

They:

- Learn relationships between words and context
- Contain billions of parameters
- Generate text by predicting sequences

Capabilities include:

- Text generation
- Summarization
- Translation
- Question answering

### Model types

- GPT → generation-focused (decoder)
- BERT → understanding-focused (encoder)
- BART / T5 → encoder-decoder (both tasks)

### Training approach

- Pretraining → large general datasets
- Fine-tuning → task-specific adaptation

This allows reuse of models without training from scratch.

## 🔹 Limitations (Hallucinations)

Generative models can produce outputs that sound correct but are incorrect.

Common issues:

- Hallucinations (false but plausible outputs)
- Bias from training data
- Sensitivity to input prompts

Mitigation requires:

- Better data
- Fine-tuning
- Human validation

## 🔹 Tools and Ecosystem

Key tools used in generative AI development:

- PyTorch → flexible deep learning framework
- TensorFlow → scalable production framework
- Hugging Face → pretrained models, tokenizers, datasets
- LangChain → building LLM-based applications
- Pydantic → data validation

---

# 📦 Module 2: Data Preparation for LLMs

## 📌 What is happening in the data pipeline

Before an LLM can process text, the data must be transformed into numerical form.

This pipeline is:

    text → tokens → indices → tensors → model input

Each step changes the representation of the data.

---

## 🔹 Tokenization

Tokenization breaks text into smaller units called tokens.

Example:

- `"IBM taught me tokenization"`  
  → `["IBM", "taught", "me", "tokenization"]`

Tokenizers vary depending on how text is split.

### Tokenization methods

#### Word-based

- Splits text into full words
- Preserves meaning
- Large vocabulary

#### Character-based

- Splits into characters
- Small vocabulary
- Longer sequences, higher computation

#### Subword-based

- Combines both approaches
- Splits rare words into smaller parts
- Keeps common words intact

Algorithms include:

- WordPiece
- Unigram
- SentencePiece

This allows better handling of unknown words.

### Tokenization → Numerical representation

After tokenization:

- Vocabulary is built
- Tokens are mapped to indices
- Unknown tokens → `<unk>`

The model ultimately receives numerical sequences, not text.

### Special tokens and padding

To structure sequences:

- `<bos>` → beginning
- `<eos>` → end
- `<pad>` → equal length

Padding ensures all sequences in a batch have the same shape.

## 🔹 Data Loaders

Data loaders handle how data is delivered to the model.

They:

- Batch data
- Shuffle samples
- Load data efficiently

This prevents memory issues and improves training performance.

### Dataset structure

Data is organized into:

- Training set
- Validation set
- Test set

Each serves a different role in training and evaluation.

### Dataset + DataLoader interaction

- Dataset → defines how data is accessed
- DataLoader → retrieves batches
- Iterator → loops over batches

This enables efficient training workflows.

### Batch processing

During batching, transformations are applied:

- Tokenization
- Index conversion
- Padding
- Tensor conversion

### Collate function

Defines how samples are combined into batches.

It:

- Processes each sample
- Applies transformations
- Returns a tensor batch

### Full data pipeline

    Dataset → DataLoader → Collate Function → Model

Each component plays a specific role in preparing data for training.

## 🔹 Data Quality and Diversity

Model performance depends heavily on training data.

### Data quality

- Remove noise and irrelevant data
- Ensure consistency
- Maintain accurate labels

Poor data → poor model performance.

### Data diversity

- Include different demographics and perspectives
- Use multiple data sources
- Cover linguistic and regional variations

This improves generalization and reduces bias.

### Data updates

- Language evolves over time
- New terms and contexts appear
- Models must be updated to stay relevant

### Ethical considerations

- Protect user privacy
- Ensure fair representation
- Maintain transparency

These are essential for building reliable AI systems.

---

## 🎯 Outcomes

This course establishes the foundation required to move into advanced topics such as transformer internals, fine-tuning, and building real-world LLM applications.
