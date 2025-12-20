import numpy as np

from .tensor import Tensor


def softmax(logits, axis=1):
    """Numerically stable softmax that keeps gradient."""
    shift = logits.data - logits.data.max(axis=axis, keepdims=True)
    exp = np.exp(shift)
    probs = exp / exp.sum(axis=axis, keepdims=True)
    out = Tensor(probs, requires_grad=logits.requires_grad)

    def _backward():
        if out.grad is None or not logits.requires_grad:
            return
        grad_input = np.empty_like(logits.data)
        for i in range(probs.shape[0]):
            p = probs[i].reshape(-1, 1)
            jac = np.diagflat(p) - p @ p.T
            grad_input[i] = jac @ out.grad[i]
        logits.grad = logits.grad + grad_input

    out._backward = _backward
    out._prev = [logits]
    return out


def cross_entropy(logits, target):
    """
    Cross entropy loss between logits (Tensor) and target class indices (array-like).
    """
    target = np.array(target, dtype=np.int64)
    shift = logits.data - logits.data.max(axis=1, keepdims=True)
    exp = np.exp(shift)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.data.shape[0]
    loss_value = -np.log(probs[np.arange(n), target] + 1e-12).mean()
    out = Tensor(loss_value, requires_grad=logits.requires_grad)

    def _backward():
        if out.grad is None or not logits.requires_grad:
            return
        grad = probs
        grad[np.arange(n), target] -= 1
        grad = grad / n
        grad = grad * out.grad
        logits.grad = logits.grad + grad if logits.grad is not None else grad

    out._backward = _backward
    out._prev = [logits]
    return out


def mse_loss(prediction, target):
    target = _ensure_array(target, prediction.data.shape)
    diff = prediction - target
    return (diff * diff).mean()


def _ensure_array(value, shape):
    arr = np.array(value, dtype=np.float64)
    if arr.shape != shape:
        arr = np.broadcast_to(arr, shape)
    return arr
