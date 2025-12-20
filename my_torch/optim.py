import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, weight_decay=0.0, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self._velocity = {id(p): np.zeros_like(p.data) for p in self.params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            if self.momentum:
                v = self._velocity[id(p)]
                v = self.momentum * v + grad
                self._velocity[id(p)] = v
                grad = v
            p.data = p.data - self.lr * grad
