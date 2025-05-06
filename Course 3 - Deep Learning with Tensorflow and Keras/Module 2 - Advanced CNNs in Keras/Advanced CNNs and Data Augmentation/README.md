# 📦 Module 2: Advanced CNNs in Keras

## Section 1: Advanced CNNs and Data Augmentation

This module focuses on implementing advanced convolutional neural networks (CNNs) and enhancing model performance through data augmentation techniques using Keras.

---

## 🧠 Advanced CNN Architectures

CNNs are designed to process visual data by stacking layers that progressively extract, downsample, and interpret features.

### 🔸 Core Layer Types:

- **Convolutional Layers** – Extract patterns using filters.
- **Pooling Layers** – Downsample feature maps.
- **Fully Connected Layers** – Final interpretation for classification.

### ⚙️ VGG-Like Architectures

VGG networks are known for:

- **Simplicity**: Use of uniform 3×3 filters
- **Depth**: Deep stacking of convolutional blocks
- **Pattern Recognition**: Effective at hierarchical feature learning

**Typical structure**:

- Repeated blocks of:
  - Two Conv2D layers (e.g., 64, 128, 256 filters)
  - MaxPooling layers
- Followed by Dense layers (e.g., 512 units)
- Final softmax output for classification

### ⚙️ ResNet-Like Architectures

ResNet (Residual Network) introduces:

- **Residual Blocks** with **shortcut connections**
- Solves the **vanishing gradient problem**
- Allows training of **very deep networks**

**Key benefits**:

- Improves convergence and gradient flow
- Allows identity mapping through skip connections

**Implementation**:

- Conv2D + BatchNorm + ReLU
- Shortcut paths added to output
- Stacked residual blocks build deep architectures

---

## 🧪 Data Augmentation Techniques

Data augmentation improves model generalization by creating modified versions of existing training images. This helps models become robust against variations and reduces overfitting.

### 🎯 Goals:

- Simulate real-world conditions
- Introduce variability without additional data
- Expose models to rotated, flipped, or distorted images

### 🔹 Basic Transformations (via `ImageDataGenerator`)

- `rotation_range`: Random rotations
- `width_shift_range`, `height_shift_range`: Translations
- `shear_range`: Geometric shearing
- `zoom_range`: Zoom in/out
- `horizontal_flip`: Flip images horizontally
- `rescale`: Normalize pixel values

```python
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True
)
```

### 🔹Normalization-Based Augmentation

You can normalize images globally or per sample:

`featurewise_center=True`: Zero mean across dataset

`featurewise_std_normalization=True`: Unit std across dataset

`samplewise_center=True`: Zero mean per image

`samplewise_std_normalization=True`: Unit std per image

```python
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    samplewise_center=True,
    samplewise_std_normalization=True
)
datagen.fit(training_images)
```

### 🔹Custom Augmentation Functions

Define and inject custom preprocessing logic:

```python
def add_random_noise(image):
    noise = np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    return image + noise

datagen = ImageDataGenerator(preprocessing_function=add_random_noise)
```

---

## ✅ Key Takeaways

#### CNN Architecture

- CNNs process visual data via convolutional, pooling, and dense layers.
- VGG uses deep stacks of small filters for structured feature extraction.
- ResNet solves training challenges with residual connections for deeper networks.

#### Data Augmentation

- Enhances generalization and reduces overfitting.
- Keras provides built-in support for augmentations via ImageDataGenerator.
- Custom functions allow fine-grained control over preprocessing.
