# Executables

## my_torch_analyzer
Chess-specific CLI that trains or predicts chessboard states from FEN input. It relies on the `chess/` module for FEN parsing and labels.

Usage:
```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

Modes:
- `--train`: trains the network using `CHESSFILE` (FEN + label).
- `--predict`: predicts labels for each FEN in `CHESSFILE`.
- `--save`: saves the trained network (train mode only).

Input format:
- Train file: each line = FEN + label
- Predict file: each line = FEN (label optional)

## my_torch_generator
Generates random networks from config files.

Usage:
```
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```

Config format:
```
input_dim=774
num_classes=5
hidden=128,64
dropout_p=0.0
```
