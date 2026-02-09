# A Frequency-Aware Hybrid CNN–Transformer for Image Manipulation Localization

## Overview

This work investigates controlled architectural modifications to a hybrid CNN–Transformer image manipulation localization framework under the standardized Protocol-CAT setting. The overall framework of this project follows [Mesorch](https://github.com/scu-zjz/Mesorch), replacing the Transformer backbone network with a CSWin-based architecture and introducing the Dice loss function to improve localization performance.


## Dataset
Training and evaluation splits are defined using JSON configuration files following the IMDLBenCo (IMDL) dataset format:
- `balanced_dataset.json` specifies the training datasets.
- `test_datasets.json` specifies the test datasets.

## Environment

- Python 3.10.0
- PyTorch 2.5.1 (CUDA 12.4, cu124)
- numpy 1.26.4
- IMDLBenCo
- git clone https://github.com/microsoft/CSWin-Transformer and adopt cswin_base_224.pth

## Training

bash training.sh

## Testing

bash testing.sh
