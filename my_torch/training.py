import numpy as np

from .functional import cross_entropy
from .optim import SGD
from .tensor import Tensor


def train(
    model,
    dataset,
    epochs=5,
    batch_size=64,
    lr=0.01,
    weight_decay=0.0,
    momentum=0.0,
    lr_decay=1.0,
    verbose=False,
):
    model.train()
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    losses = []
    current_lr = lr
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for x, y in dataset.batches(batch_size, shuffle=True):
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.data)
            batches += 1
        mean_loss = epoch_loss / batches if batches else 0.0
        losses.append(mean_loss)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {mean_loss:.4f}")
        if lr_decay != 1.0:
            current_lr *= lr_decay
            optimizer.lr = current_lr
    return losses


def predict(model, features, batch_size=256):
    model.eval()
    predictions = []
    probabilities = []
    for start in range(0, len(features), batch_size):
        batch = Tensor(features[start : start + batch_size], requires_grad=False)
        logits = model(batch)
        probs = _softmax_np(logits.data)
        probabilities.append(probs)
        predictions.append(probs.argmax(axis=1))
    if not predictions:
        return np.array([]), np.empty((0, 0))
    return np.concatenate(predictions), np.vstack(probabilities)


def evaluate_accuracy(model, dataset, batch_size=256):
    model.eval()
    total = 0
    correct = 0
    for start in range(0, len(dataset.features), batch_size):
        batch = Tensor(dataset.features[start : start + batch_size], requires_grad=False)
        targets = dataset.labels[start : start + batch_size]
        logits = model(batch)
        preds = logits.data.argmax(axis=1)
        correct += (preds == targets).sum()
        total += len(targets)
    if total:
        return correct / total
    else:
        return 0.0


def _softmax_np(x):
    shift = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shift)
    return exp / exp.sum(axis=1, keepdims=True)
