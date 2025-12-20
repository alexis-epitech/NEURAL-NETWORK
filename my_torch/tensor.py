import numpy as np


def _to_tensor(value):
    """Ensure value is a Tensor instance."""
    return value if isinstance(value, Tensor) else Tensor(value, requires_grad=False)


def _unbroadcast(grad, shape):
    """Reduce broadcasted gradient to match original shape."""
    if grad is None:
        return None
    if shape == ():
        return np.array(grad).sum().reshape(())
    result = grad

    while len(result.shape) > len(shape):
        result = result.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            result = result.sum(axis=axis, keepdims=True)
    return result


class Tensor:
    """Minimal autograd-enabled tensor used by the my_torch library."""

    def __init__(self, data, requires_grad=False, name=None):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = []
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False, name=self.name)

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float64)
        self.grad = self.grad + grad if self.grad is not None else grad

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = _to_tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad = other.grad + _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = self.grad - out.grad

        out._backward = _backward
        out._prev = [self]
        return out

    def __mul__(self, other):
        other = _to_tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad = other.grad + _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = _to_tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad = other.grad - _unbroadcast(out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __rtruediv__(self, other):
        other = _to_tensor(other)
        return other / self

    def __matmul__(self, other):
        other = _to_tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = self.grad + out.grad @ other.data.T
            if other.requires_grad:
                other.grad = other.grad + self.data.T @ out.grad

        out._backward = _backward
        out._prev = [self, other]
        return out

    def transpose(self, axes=None):
        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = self.grad + out.grad.transpose(axes)

        out._backward = _backward
        out._prev = [self]
        return out

    @property
    def T(self):
        return self.transpose()

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad = self.grad + np.ones_like(self.data) * grad

        out._backward = _backward
        out._prev = [self]
        return out

    def mean(self, axis=None, keepdims=False):
        count = self.data.size if axis is None else np.array(self.data).shape[axis] if not isinstance(axis, (list, tuple)) else np.prod([self.data.shape[a] for a in axis])
        return self.sum(axis=axis, keepdims=keepdims) / count

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = self.grad + out.grad * (self.data > 0)

        out._backward = _backward
        out._prev = [self]
        return out

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig, requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = self.grad + out.grad * sig * (1 - sig)

        out._backward = _backward
        out._prev = [self]
        return out

    def __repr__(self):
        name = f"{self.name}=" if self.name else ""
        return f"Tensor({name}{self.data}, requires_grad={self.requires_grad})"
