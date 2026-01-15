# 📘 Module 3: Linear Regression PyTorch Way

## 📦 Section 2: Optimization in PyTorch

This section introduces how model parameters are updated using PyTorch's optimizer system. The training process is structured around a standardized workflow for building and optimizing models using PyTorch modules, autograd, and built-in optimizers.

## 🧠 Topics and Concepts

### Optimizer Setup in PyTorch

A PyTorch optimizer manages the learning parameters of a model and applies updates using the gradient information. To begin the optimization process:

- A dataset object is created to hold training inputs and labels.
- A model class is defined as a subclass of `nn.Module`, encapsulating all learnable parameters like weights and biases.
- A criterion (loss) function is selected from `torch.nn`, typically `nn.MSELoss()` in regression tasks.
- A `DataLoader` is used to batch and iterate over the dataset efficiently.

### Constructing and Using the Optimizer

PyTorch optimizers are created using classes from `torch.optim`, such as `torch.optim.SGD`. The optimizer is initialized by:

- Passing the model parameters using `model.parameters()`.
- Setting a learning rate that determines the size of the update step per iteration.

Each optimizer instance keeps an internal state dictionary accessible via `optimizer.state_dict()`, storing hyperparameters and the current optimization state.

### Optimization Workflow

Training is performed over multiple epochs. Each epoch includes:

1. **Batch iteration**: Iterating over data batches from the `DataLoader`.
2. **Forward pass**: Using the model to predict outputs from the input.
3. **Loss computation**: Measuring the prediction error using the criterion.
4. **Gradient reset**: Calling `optimizer.zero_grad()` to clear previous gradients.
5. **Backpropagation**: Running `loss.backward()` to compute new gradients.
6. **Parameter update**: Applying `optimizer.step()` to adjust weights and biases.

This process is repeated for each batch across all epochs, progressively minimizing the loss.

### Conceptual Flow

Even though the optimizer and loss function are not directly linked in code, PyTorch’s autograd engine constructs a computational graph that automatically connects these components during training:

- The optimizer is initialized with the model parameters.
- The model computes predictions `ŷ` from inputs `x`.
- The loss function evaluates prediction error.
- `loss.backward()` computes gradients.
- `optimizer.step()` updates the model parameters accordingly.

This abstraction makes it easy to scale the workflow to more complex architectures and optimizers.

## ✅ Takeaways

- PyTorch optimizers like SGD manage gradient descent updates over model parameters.
- `optimizer.step()` is the key method for applying parameter changes.
- A typical training loop involves forward pass, loss calculation, backpropagation, and parameter update.
- PyTorch’s autograd system handles the computational graph and gradients automatically.
- The same workflow generalizes to complex models and advanced optimizers.
