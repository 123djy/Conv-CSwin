# A Frequency-Aware Hybrid CNN–Transformer for Image Manipulation Localization

## Overview

This work investigates controlled architectural modifications to a hybrid CNN–Transformer image manipulation localization framework under the standardized Protocol-CAT setting. The overall framework of this project follows [Mesorch](https://github.com/scu-zjz/Mesorch}, replacing the Transformer backbone network with a CSWin-based architecture and introducing the Dice loss function to improve localization performance.


## Dataset
Training and evaluation splits are defined using JSON configuration files following the IMDLBenCo (IMDL) dataset format:
- `balanced_dataset.json` specifies the training datasets.
- `test_datasets.json` specifies the test datasets.
