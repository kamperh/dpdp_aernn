# DPDP Autoencoding Recurrent Neural Network (DPDP AE-RNN)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](license.md)


## Overview

The segmental AE-RNN is described in

> To-do: write 
> the paper

Please cite this paper if you use the code.


## Examples

To run the algorithm on the [Brent corpus](https://arxiv.org/abs/cs/9905007),
execute:

    ./run_brent.py

Step-by-step examples are given in the Jupyter notebooks in the
[notebooks/](notebooks/) directory. The [brent.ipynb](notebooks/brent.ipynb)
notebook follows the script above the most closely. The other notebooks rely on
encodings obtained from other models.


## Brent results

The example script above is by default applied to a validation set (`sentences
= val_sentences_ref`) on which you should obtain the following results:

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 79.8904%
    Recall: 84.9014%
    F-score: 82.3197%
    OS: 6.2724%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 69.7212%
    Recall: 72.7413%
    F-score: 71.1993%
    OS: 4.3317%
    ---------------------------------------------------------------------------

When applying the model on the full training dataset (`sentences =
train_sentences_ref`), you should obtain the following results:

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 77.5189%
    Recall: 84.6965%
    F-score: 80.9489%
    OS: 9.2592%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 66.4072%
    Recall: 70.7536%
    F-score: 68.5115%
    OS: 6.5451%
    ---------------------------------------------------------------------------


## Acknowledgements

The Brent data was obtained from
<https://github.com/melsner/neural-segmentation>.