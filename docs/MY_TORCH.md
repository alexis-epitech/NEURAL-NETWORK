# my_torch Library Guide

## Overview
`my_torch` is a lightweight neural-network library inspired by PyTorch. It is designed to be reusable for any small supervised-learning task.

The library provides:
- `Tensor` with autograd
- Layers: `Linear`, `ReLU`, `Sigmoid`, `Dropout`
- `Sequential` container
- `SGD` optimizer
- Loss functions: `cross_entropy`, `mse_loss`
- Model save/load utilities

## Getting Started
Requirements:
- Python 3
- NumPy

### Create a model:
```python
from my_torch import nn

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 3),
)
```

### Train a model:
```python
from my_torch import training

training.train(model, dataset, epochs=10, batch_size=64, lr=1e-2, verbose=True)
```

### Save a model:
```python
from my_torch import io

io.save_model(model, "my_torch_network_basic.nn", metadata={"input_dim": 10, "num_classes": 3})
```

### Load a model:
```python
from my_torch import io

model, metadata = io.load_model("my_torch_network_basic.nn")
```

### Predict:
```python
from my_torch import training

preds, _ = training.predict(model, features)
print(preds[:5])
```

## Library Structure
- `my_torch/tensor.py`: `Tensor` and autograd operations
- `my_torch/nn.py`: `Module`, layers, and `Sequential`
- `my_torch/optim.py`: `SGD` optimizer
- `my_torch/functional.py`: loss functions and softmax
- `my_torch/training.py`: training and prediction helpers
- `my_torch/io.py`: model serialization

## Core Concepts
### Tensor and Autograd
`Tensor` wraps a NumPy array and tracks gradients when `requires_grad=True`. Each operation creates a node in a computation graph. Calling `backward()` on a scalar loss computes gradients for all parameters connected to that graph. Gradients accumulate by default, so call `zero_grad()` before each update step.

### Modules
All layers inherit from `Module`. A module owns trainable parameters and can contain other modules. `Module.parameters()` returns a flat list of all parameters, and `train()` / `eval()` propagate the mode to child modules (important for Dropout).

## Layers
`Linear` is a fully connected layer that applies `y = xW^T + b` with trainable weights and bias.  
`ReLU` applies `max(0, x)` element-wise to introduce non-linearity.  
`Sigmoid` squashes values to the range `[0, 1]`, useful for binary outputs or gating.  
`Dropout` randomly zeroes activations during training and scales the rest. It is disabled during evaluation (`model.eval()`).

## Sequential Container
`Sequential` chains layers in order, passing the output of each layer to the next. It is the fastest way to build a model without writing a custom class. Parameters are collected recursively from all children.

## SGD Optimizer
`SGD` performs gradient descent updates: `param = param - lr * grad`.  
It supports:
- `momentum`: smooths updates across steps
- `weight_decay`: L2 regularization

The standard loop is:
1) forward pass  
2) compute loss  
3) `loss.backward()`  
4) `optimizer.step()`  

## Loss Functions
- `cross_entropy(logits, target)`: for multi-class classification, `target` is class indices.
- `mse_loss(prediction, target)`: for regression or matching continuous targets.

## Training Utilities
`training.train` expects a dataset that yields batches of `(Tensor, labels)`. It handles the training loop, backprop, and optimizer steps.  
`training.predict` runs inference and returns predicted class indices and probabilities.  
`training.evaluate_accuracy` returns the accuracy on a labeled dataset.

## Model Save/Load
`io.save_model` serializes the model architecture and weights to JSON, plus optional metadata.  
`io.load_model` reconstructs the model and returns `(model, metadata)`.
