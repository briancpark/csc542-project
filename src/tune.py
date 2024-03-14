"""Hyperparameter tuning for the model"""

import json
import ray
from src.training import finetuning


def hpo_tune(model_path, tokenizer_path, train_dataset_name):
    """Hyperparameter tuning. Mainly performs a grid search"""

    with open("config/param_grid.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    total_search_space = 1
    for param in params:
        total_search_space *= len(params[param])

    print(f"The total search space is {total_search_space}")

    ray.init()

    r = ray.remote(num_gpus=1)
    ray_finetune = r(finetuning)

    oids = []

    for rank in params["rank"]:
        for alpha in params["alpha"]:
            for layers in params["layers"]:
                for dropout in params["dropout"]:
                    try:
                        oids.append(
                            ray_finetune.remote(
                                model_path,
                                tokenizer_path,
                                train_dataset_name,
                                epochs=1,
                                batch_size=1,
                                rank=rank,
                                alpha=alpha,
                                layers=layers,
                                dropout=dropout,
                            )
                        )
                    # pylint: disable=broad-except
                    except Exception as e:
                        print(e)

    results = ray.get(oids)

    results.sort(reverse=True)
    print(results[:5])

    ray.shutdown()
