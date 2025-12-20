import json
import os

from . import nn


def serialize_module(module):
    return module.state_dict()


def deserialize_module(state):
    layer_type = state.get("type")
    if layer_type == "Sequential":
        layers = [deserialize_module(s) for s in state["layers"]]
        return nn.Sequential(*layers)
    if layer_type == "Linear":
        layer = nn.Linear(state["in_features"], state["out_features"])
        layer.load_state_dict(state)
        return layer
    if layer_type == "ReLU":
        return nn.ReLU()
    if layer_type == "Sigmoid":
        return nn.Sigmoid()
    if layer_type == "Dropout":
        layer = nn.Dropout(state.get("p", 0.5))
        return layer
    raise ValueError(f"Unknown layer type: {layer_type}")


def save_model(model, path, metadata=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "architecture": serialize_module(model),
        "metadata": metadata or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_model(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = deserialize_module(data["architecture"])
    metadata = data.get("metadata", {})
    return model, metadata
