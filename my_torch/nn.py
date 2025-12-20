import numpy as np

from .tensor import Tensor


class Module:
    """Base class for all network modules."""

    def parameters(self):
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor):
                if value.requires_grad:
                    params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self.training = True
        for child in self._children():
            child.train()

    def eval(self):
        self.training = False
        for child in self._children():
            child.eval()

    def _children(self):
        children = []
        for value in self.__dict__.values():
            if isinstance(value, Module):
                children.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        children.append(item)
        return children

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        limit = np.sqrt(6 / (in_features + out_features))
        weight_init = np.random.uniform(-limit, limit, (out_features, in_features))
        self.weight = Tensor(weight_init, requires_grad=True, name="weight")
        self.bias = Tensor(np.zeros(out_features), requires_grad=True, name="bias")
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return x @ self.weight.transpose() + self.bias

    def state_dict(self):
        return {
            "type": "Linear",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight": self.weight.data.tolist(),
            "bias": self.bias.data.tolist(),
        }

    def load_state_dict(self, state):
        self.weight.data = np.array(state["weight"], dtype=np.float64)
        self.bias.data = np.array(state["bias"], dtype=np.float64)

    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features})"


class ReLU(Module):
    def forward(self, x):
        return x.relu()

    def state_dict(self):
        return {"type": "ReLU"}

    def load_state_dict(self, state):
        return self

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

    def state_dict(self):
        return {"type": "Sigmoid"}

    def load_state_dict(self, state):
        return self

    def __repr__(self):
        return "Sigmoid()"


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = float(p)
        self.training = True
        self._mask = None

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (np.random.rand(*x.data.shape) < keep_prob).astype(np.float64) / max(keep_prob, 1e-8)
        self._mask = mask
        out = Tensor(x.data * mask, requires_grad=x.requires_grad)

        def _backward():
            if out.grad is None or not x.requires_grad:
                return
            x.grad = x.grad + out.grad * mask

        out._backward = _backward
        out._prev = [x]
        return out

    def state_dict(self):
        return {"type": "Dropout", "p": self.p}

    def load_state_dict(self, state):
        self.p = float(state.get("p", self.p))
        return self

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        self.training = True

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def append(self, layer):
        self.layers.append(layer)

    def state_dict(self):
        return {
            "type": "Sequential",
            "layers": [layer.state_dict() for layer in self.layers],
        }

    def load_state_dict(self, state):
        for layer, layer_state in zip(self.layers, state["layers"]):
            layer.load_state_dict(layer_state)

    def __repr__(self):
        return "Sequential(\n  " + "\n  ".join(repr(l) for l in self.layers) + "\n)"
