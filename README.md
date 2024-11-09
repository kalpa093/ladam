# Lightweight Attention-based Data Augmentation Method (LADAM)

This repository contains the code for our paper All You Need is Attention: Lightweight Attention-based Data Augmentation Augmentation Method (EMNLP 2024 Findings).

![Overview_Crossover_1](https://github.com/user-attachments/assets/d40e024d-0165-4ccc-8682-b05abda83cb6)

# Overview

1. We introduces LADAM, a novel method for enhancing the performance of text classification tasks. LADAM employs attention scores to exchange semantically similar words between sentences.
2. Our experimental results across five datasets demonstrate that LADAM consistently outperforms other baseline methods across diverse conditions.

# Implementations
## Environments

    git clone https://github.com/kalpa093/ladam.git
    cd ladam

    pip install pip install torch==2.1.1+cu121 torchvision==0.15.2+cu121 torchaudio==2.1.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    
## Use LADAM

    # Execute LADAM for text augmentation
    python ladam.py -d dataset -m model
    
    # Training
    python train.py -d dataset -n n_aug

`-d` : `dataset` should be file name of dataset (e.g. CR.csv)

`-m` : `model` should be name of model (e.g. bert, roberta, deberta, distilbert)

`-n` : `n_aug` is a number of augmentation (e.g. 1, 2, 4, ... )


# Contact

If you have any questions abiout the code or the paper, feel free to open an issue.
