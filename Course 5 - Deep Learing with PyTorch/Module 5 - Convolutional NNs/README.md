# 📌 Module 5: Convolutional Neural Networks (CNNs)

This module focuses on **Convolutional Neural Networks (CNNs)** and their role in learning structured representations from image data.

These different modules explore how CNNs are constructed, trained, and extended using PyTorch. It progresses from the fundamental convolution operation to full CNN architectures, transfer learning with pre-trained models, and hardware acceleration using GPUs.

---

## 🔹 Convolution as a Core Operation

Convolution is introduced as a solution to the loss of spatial information that occurs when images are flattened into vectors.  
Instead of relying on absolute pixel positions, convolution processes **local neighborhoods** using learnable kernels that slide across the image.

Key concepts learned include:

- Kernels as learnable parameter matrices
- Activation maps as the output of convolution
- Bias terms broadcast across spatial dimensions
- Preservation of spatial relationships through local receptive fields

The size of the activation map is determined by:

- Input dimensions
- Kernel size
- Stride
- Zero padding

Zero padding is used to control spatial resolution and allow valid convolution when larger strides or kernels are applied.

## 🔹 Activation Functions and Max Pooling

After convolution, **activation functions** are applied elementwise to activation maps.  
This introduces nonlinearity while preserving tensor shape and channel structure.

Max pooling is then applied as a **downsampling operation**, reducing spatial dimensions while retaining the strongest local responses.

Concepts emphasized:

- Activation functions operate on activation maps, not raw images
- Pooling is applied channel-wise
- Pooling reduces parameter count and improves robustness to small spatial shifts
- Pooling output dimensions follow the same reasoning as convolution output sizes

## 🔹 Multiple Input and Output Channels

The module explains how convolution generalizes to:

- Multiple output channels (multiple kernels per layer)
- Multiple input channels (one kernel per input channel, summed)
- Combined multi-input, multi-output convolution

Each output channel:

- Has a dedicated set of kernels (one per input channel)
- Produces a distinct activation map
- Extracts a different feature representation

This structure enables CNNs to combine information across channels and learn increasingly expressive visual features.

## 🔹 Convolutional Neural Network Architecture

A CNN is presented as a structured pipeline composed of:

- Convolution layers with learnable kernels
- Activation functions
- Pooling layers
- Flattening operations
- Fully connected output layers

Key architectural ideas:

- Convolution and pooling stages progressively extract and compress features
- Flattening bridges spatial feature maps to dense classification layers
- Output layer size corresponds to the number of target classes
- Calculating intermediate tensor shapes is a critical design step

The forward pass defines the exact sequence of operations, while training updates all convolutional and linear parameters via backpropagation.

## 🔹 Transfer Learning with TorchVision Models

The module introduces **transfer learning** using pre-trained models from TorchVision.

Core ideas:

- Expert-trained CNNs can be reused as fixed feature extractors
- Only the final classification layer is replaced and retrained
- All other parameters are frozen to prevent unnecessary updates

The process includes:

- Loading a pre-trained model
- Replacing the output layer to match the new task
- Training only the new layer on a custom dataset

This approach reduces training time and improves performance when data is limited.

## 🔹 GPUs and Accelerated Training in PyTorch

The module concludes by explaining how to use **GPUs** to accelerate CNN training.

Key points:

- CUDA enables GPU computation in PyTorch
- Tensors and models must be explicitly transferred to the GPU
- Model architecture and forward logic remain unchanged
- Training requires both inputs and labels on the GPU
- Evaluation requires only input data on the GPU

GPU usage significantly improves performance for computationally intensive CNN workloads.

---

## ✅ Key Takeaways

- Convolution preserves spatial structure and enables local feature learning
- Activation and pooling operate on activation maps, not raw inputs
- Multi-channel convolution enables rich feature extraction
- CNNs combine convolution, pooling, flattening, and dense layers into a unified pipeline
- Transfer learning allows reuse of expert-trained models with minimal retraining
- GPUs provide substantial speedups without changing model logic

This module establishes the foundational principles required to design, train, and deploy convolutional neural networks in PyTorch.
