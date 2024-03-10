"""Hyperparameter tuning for the model"""

import os
import json
import torch
import ray
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.utils import device, load_model, allocated_memory, torch_timer
from src.inference import dataset_inference
from src.training import finetuning


def hpo_tune(model_path, tokenizer_path, train_dataset_name, test_dataset_name):
    """Hyperparameter tuning. Mainly performs a grid search"""

    with open("config/param_grid.json", "r") as f:
        params = json.load(f)

    total_search_space = 1
    for param in params:
        total_search_space *= len(params[param])

    print(f"The total search space is {total_search_space}")

    ray.init()

    r = ray.remote(num_gpus=1, num_returns=2)
    ray_finetune = r(finetuning)

    oids = []

    for rank in params["rank"]:
        for alpha in params["alpha"]:
            for layers in params["layers"]:
                for dropout in params["dropout"]:
                    oids.append(
                        ray_finetune.remote(
                            model_path,
                            tokenizer_path,
                            train_dataset_name,
                            epochs=30,
                            batch_size=2,
                            rank=rank,
                            alpha=alpha,
                            layers=layers,
                            dropout=dropout,
                        )
                    )

    results = ray.get(oids)

    results.sort(reverse=True)
    print(results[:5])

    ray.shutdown()
