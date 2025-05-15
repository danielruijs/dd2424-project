# Comparing RNN, LSTM, and Transformer Architectures for Character-Level Generation of Harry Potter Text

This is a project in the course DD2424 Deep Learning in Data Science at KTH Royal Institute of Technology. The goal of this project is to compare the performance of different neural network architectures (RNN, LSTM, and Transformer) for character-level generation of text from the Harry Potter series.

## Packages

[UV](https://docs.astral.sh/uv/getting-started/) is used for this project. To add packages, run the following command in the terminal:
```bash
uv add <package_name>
```
To remove packages, run the following command in the terminal:
```bash
uv remove <package_name>
```
To run a the scripts in this repository run:
```bash
uv sync
```
Then activate the virtual environment with:
```bash
# On Windows
source .venv/Scripts/activate
# On Linux/MacOS
source .venv/bin/activate
```

## Training

To train the models, run the following command:
```bash
sh train.sh <model-to-train>
```
The following models are available: "rnn", "lstm", "lstm2", and "transformer". To change model and training parameters, change the constants in the `main.py` file.

### Logging

Logs are saved in the `logs` directory. You can monitor the training process using TensorBoard. To do this, run the following command in a separate terminal:

```bash
tensorboard --logdir logs/ --port 6006
```

Then open your web browser and go to http://localhost:6006 to view the TensorBoard dashboard.

## Hyperparameter tuning

The `hp_tuning` directory contains scripts for hyperparameter tuning. Specify the parameters to tune in the `hp_tuning.py` file and start hyperparameter tuning with the following command:
```bash
sh hp_tuning.sh
```
Logs will be saved to a separate folder and results can be visualized using TensorBoard with the command above.

## Tokenization

The `tokenization` directory contains training scripts for training the models with Byte-Pair Encoding and Word2Vec tokenization. Training is performed with the same command as above.
