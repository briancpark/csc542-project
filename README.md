# CSC/ECE 542 Project

Brian Park, Fangyuan (Sophie) Cheng, Tzu-Ching (Scott) Yeh

This is the final course project for CSC/ECE 542: Neural Networks.

## Setup

First create a conda environment. We use Python 3.10:

```sh
conda create -n csc542 python=3.10
```

As always, activate the environment and install the dependencies:

```sh
conda activate csc542
pip install -r requirements.txt
```

## Running the Code

```sh
python3 main.py -h  # for help
python3 main.py --inference # for inference
python3 main.py --inference --test-dataset # for testing the model
python3 main.py --finetuning # for finetuning/training with LoRA
```
