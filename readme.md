# Machine Translation: English to Hindi Using Transformers

This project implements a transformer-based neural network architecture from scratch using PyTorch for machine translation between English and Hindi. The model is trained on the IIT Bombay English-Hindi Parallel Corpus to translate English sentences into Hindi.

![alt text](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Results](#results)
- [Further Improvements](# Further-Improvements)
- [References](#references)

## Introduction

Machine Translation (MT) is the task of automatically converting text from one language to another. This project focuses on building a transformer-based model to translate English sentences into Hindi. Transformers have become the standard architecture for most NLP tasks, including machine translation, due to their efficiency in capturing dependencies in sequential data.

## Dataset

We use the **IIT Bombay English-Hindi Parallel Corpus**, which contains around **1.6 million** sentence pairs with their corresponding translations in both languages. The dataset consists of formal translations in various domains such as health, tourism, and general topics.

- **Source Language**: English
- **Target Language**: Hindi
- **Dataset Link**: [IIT Bombay English-Hindi Corpus](http://www.cfilt.iitb.ac.in/iitb_parallel/)

### Preprocessing

- Tokenization of both English and Hindi sentences using word-level tokenizers.
- Sequence padding and truncation to handle varying sentence lengths.
- Vocabulary creation and encoding for both languages.

## Model Architecture

The project implements the **Transformer** model from scratch in PyTorch. Key components of the transformer architecture include:

1. **Multi-Head Attention**: Enables the model to attend to different parts of the sequence simultaneously. Used in the form of Masked Multi Head Attention and Cross Attention in Decoder through slight modification.
2. **Positional Encoding**: Adds information about the position of words in the sequence.
3. **Feed-Forward Networks**: Non-linear transformation of attention output.
4. **Layer Normalization**: Stabilizes the training and speeds up convergence.

### Model Details

- **Number of Layers**: 6 layers for both encoder and decoder.
- **Heads in Attention Mechanism**: 8
- **Hidden Size**: 512
- **Dropout**: 0.1 for regularization.
- **Optimizer**: Adam 

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+
- TorchText
- Numpy
- tensorboard
- tokenizers
- Datasets

To install the necessary packages, run:

```bash
pip install -r requirements.txt
```

## Results

The model was evaluated using the Character Error Rate, Word Error Rate and BLEU score to measure translation accuracy. 

After training, the model achieved : 
- **Character Error Rate:** 0.3548728823661804
- **Word Error Rate:** 0.6785714030265808
- **BLEU Score:** 0.0

Example translations:
Input: "I love machine learning."

Predicted Translation: "मुझे मशीन लर्निंग पसंद है।"

## Further-Improvements

- The following model was trained till 20 epochs. The performance can be increased by improving the number of epochs and introducing early stopping so that our model does not overfit
- As the dataset is huge only 10% of the data is used for training and validation due to limited resources. The performance can be further improved by using the entire dataset.

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [IIT Bombay English-Hindi Corpus](https://www.cfilt.iitb.ac.in/iitb_parallel/)