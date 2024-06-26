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
python3 main.py -h                                  # for help
python3 main.py --inference                         # for inference on a single prompt
python3 main.py --inference -p \ 
        "Question: Write 3sum in Python: Answer:\n" # for inference on a custom prompt
python3 main.py --inference-evaluate                # for evaluating inference on the whole dataset
python3 main.py --finetuning                        # for finetuning/training with LoRA
python3 main.py --eda                               # for exploratory data analysis
python3 main.py --hyperparameter-tune               # for hyperparameter tuning
```


## Model Card

| Model Name | Layers | Heads | Hidden Size |
|------------|--------|-------|-------------|
| LLaMA-1.1B | 22     | 32    | 2048        |
| LLaMA-7B   | 32     | 32    | 4096        |
