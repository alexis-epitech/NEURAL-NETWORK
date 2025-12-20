import numpy as np

from .tensor import Tensor

PIECE_TO_INDEX = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
FEATURE_DIM = 64 * len(PIECE_TO_INDEX) + 1 + 4 + 1


def fen_to_vector(fen):
    parts = fen.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Invalid FEN string: {fen}")
    board, active, castling, en_passant = parts[:4]
    encoded_board = []
    for rank in board.split("/"):
        for char in rank:
            if char.isdigit():
                encoded_board.extend([0] * int(char) * len(PIECE_TO_INDEX))
            else:
                one_hot = [0] * len(PIECE_TO_INDEX)
                idx = PIECE_TO_INDEX.get(char)
                if idx is None:
                    raise ValueError(f"Unexpected piece '{char}' in FEN '{fen}'")
                one_hot[idx] = 1
                encoded_board.extend(one_hot)
    if len(encoded_board) < 64 * len(PIECE_TO_INDEX):
        encoded_board.extend([0] * (64 * len(PIECE_TO_INDEX) - len(encoded_board)))
    active_flag = 1.0 if active == "w" else 0.0
    castling_flags = [
        1.0 if "K" in castling else 0.0,
        1.0 if "Q" in castling else 0.0,
        1.0 if "k" in castling else 0.0,
        1.0 if "q" in castling else 0.0,
    ]
    en_passant_flag = 0.0 if en_passant == "-" else 1.0
    vector = np.array(encoded_board + [active_flag] + castling_flags + [en_passant_flag], dtype=np.float64)
    return vector


def parse_chess_file(path, label_to_index=None, require_label=True):
    label_map = label_to_index or LABEL_TO_INDEX
    features = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            fen_fields = parts[:6]
            label_str = " ".join(parts[6:]) if len(parts) > 6 else None
            if label_str:
                if label_str not in label_map:
                    raise ValueError(f"Unknown label '{label_str}' in {path}")
                labels.append(label_map[label_str])
            elif require_label:
                raise ValueError(f"Missing label in line: {line}")
            features.append(fen_to_vector(" ".join(fen_fields)))
    if not features:
        raise ValueError(f"No data found in {path}")
    label_array = None
    if labels:
        label_array = np.array(labels, dtype=np.int64)
    return np.vstack(features), label_array


class ChessDataset:
    def __init__(self, features, labels):
        if labels is None:
            raise ValueError("Labels are required for training dataset")
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def batches(self, batch_size, shuffle=True):
        idx = np.arange(len(self.labels))
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            x = Tensor(self.features[batch_idx], requires_grad=False)
            y = self.labels[batch_idx]
            yield x, y
