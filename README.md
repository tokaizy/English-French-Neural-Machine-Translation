# English--French Neural Machine Translation with Attention (Seq2Seq LSTM)

Neural Machine Translation system that translates **English sentences
into French** using a **Sequence-to-Sequence Encoder--Decoder LSTM
architecture implemented from scratch in PyTorch**.

------------------------------------------------------------------------

## Overview

This project implements a classical **Seq2Seq Neural Machine Translation
(NMT)** model without attention.\
The system learns to encode English sentences into a latent
representation and generate the corresponding French translation
token-by-token.

Key components:

-   Encoder--Decoder LSTM architecture
-   Custom tokenization and vocabulary building
-   Teacher forcing during training
-   Packed sequences for variable-length sentences
-   BLEU score evaluation

The model is trained using the **Multi30k dataset**, which contains
parallel English--French sentence pairs describing visual scenes.

------------------------------------------------------------------------

## Architecture

### Encoder

-   Embedding dimension: **256**
-   LSTM layers: **2**
-   Hidden size: **512**
-   Dropout: **0.3**

The encoder reads the English input sequence and compresses it into a
**context vector** using the final hidden state.

### Decoder

-   Embedding dimension: **256**
-   LSTM layers: **2**
-   Hidden size: **512**
-   Output layer projecting to **10,000 token vocabulary**

The decoder generates the French sentence **autoregressively** using the
encoder context vector.

------------------------------------------------------------------------

## Model Configuration

  Parameter               Value
  ----------------------- --------
  Embedding Dimension     256
  Hidden Size             512
  LSTM Layers             2
  Dropout                 0.3
  Vocabulary Size         10,000
  Max Sequence Length     50
  Batch Size              64
  Teacher Forcing Ratio   0.5
  Optimizer               Adam
  Learning Rate           0.001

Training uses **ReduceLROnPlateau learning rate scheduling** and **early
stopping** for improved convergence.

------------------------------------------------------------------------

## Dataset

**Multi30k Dataset**

-   Training samples: **29,000**
-   Validation samples: **1,000**
-   Test samples: **1,000**

Total: **31,000 English--French sentence pairs**.

Sentences typically contain **10--15 words** describing everyday scenes.

------------------------------------------------------------------------

## Training Pipeline

1.  Tokenize sentences using **spaCy**
2.  Build vocabulary with special tokens:

```{=html}
<!-- -->
```
    <unk> <pad> <sos> <eos>

3.  Convert sentences to token indices
4.  Pad sequences for batch training
5.  Train using **CrossEntropyLoss (ignoring padding tokens)**
6.  Apply **teacher forcing during decoding**

------------------------------------------------------------------------

## Results

The optimized training configuration significantly improved translation
quality.

  Metric   Version 1   Version 2(with Attention)
  -------- ----------- -----------
  BLEU-1   38.73       73.86
  BLEU-2   23.33       64.45
  BLEU-3   15.49       56.90
  BLEU-4   10.86       **50.81**

BLEU‑4 improved by **\~368%**, demonstrating the importance of training
optimization.

------------------------------------------------------------------------

## Example Translation

Input (English)

    two men are sitting in a canoe in the middle of a lake

Output (French)

    deux hommes sont assis dans un canoë au milieu d'un lac

------------------------------------------------------------------------

## Tech Stack

-   Python
-   PyTorch
-   spaCy
-   torchtext
-   sacreBLEU

------------------------------------------------------------------------
