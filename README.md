# Vanilla RNN for Text Generation and Next-Word Prediction

This project implements a Vanilla Recurrent Neural Network (RNN) to generate text and predict the next word in a sequence. It uses the "Tiny Shakespeare" dataset available from Hugging Face and implements two variations of the model: one with randomly initialized embeddings and another with pretrained embeddings.

## Requirements

To run this project, you will need the following dependencies:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- tqdm
- Hugging Face `datasets`
- Scikit-learn

You can install all dependencies by running:

```bash
pip install torch numpy matplotlib seaborn tqdm datasets scikit-learn
```

## Dataset
This project uses the Tiny Shakespeare dataset, which is a subset of Shakespeare's works available from Hugging Face's datasets library.

```bash
from datasets import load_dataset
dataset = load_dataset("tiny_shakespeare")
```

## Code Overview
The project consists of the following key parts:

# 1. Data Preprocessing
Tokenization: The dataset is tokenized into individual words.

Vocabulary Creation: A vocabulary is created from the unique tokens in the dataset.

Input-Output Sequences: The data is transformed into sequences of words where each sequence is used to predict the next word.

# 2. Model Architecture
Two model variations are implemented:

Vanilla RNN with Random Embeddings
Vanilla RNN with Pretrained Embeddings

Both models consist of the following layers:
Embedding Layer: Converts words into dense vectors.
RNN Layer: Processes sequences of words using a Recurrent Neural Network.
Fully Connected Layer: Outputs a prediction of the next word in the sequence.

# 3. Training
The training process consists of the following steps:
The model is trained on the input-output sequences with a CrossEntropyLoss criterion.
The Adam optimizer is used to optimize the model parameters.
Gradient clipping is applied to prevent exploding gradients.

# 4. Evaluation
After training, the following evaluation tasks are performed:
Perplexity Calculation: Measures the model's performance in predicting the next word.
Next-Word Generation: Given a seed text, the model generates a sequence of words by predicting the next word iteratively.
Confusion Matrix: Visualizes the model's performance using a confusion matrix.

# 5. Loss and Accuracy Visualization
The training loss is plotted over epochs to visualize the convergence of the model during training. Additionally, a confusion matrix is generated to evaluate the model's predictions.

# Training
To train the model, you can run the script by executing:

bash
```
python train.py
```





