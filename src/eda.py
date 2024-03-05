"""Here lies the exploratory data analysis for the dataset"""

# import torch
from datasets import load_dataset
from tqdm import tqdm
from src.utils import device, load_model
from src.inference import evaluate_code


def get_token_counts(tokenizer, dataset):
    """Get the token counts for the dataset"""
    token_counts = {}
    for example in tqdm(dataset):
        code_solution = example["prompt"] + example["canonical_solution"]
        ids = tokenizer(code_solution, return_tensors="pt").input_ids.to(device)

        for token_id in ids[0]:
            token = tokenizer.decode(token_id, skip_special_tokens=True)
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1
    return token_counts


def eda(model_path, tokenizer_path, dataset_path):
    """Exploratory data analysis for the dataset"""
    tokenizer, _ = load_model(model_path, tokenizer_path, lora=False)

    dataset = load_dataset(dataset_path, split="test")

    token_counts = get_token_counts(tokenizer, dataset)

    for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[
        :100
    ]:
        print(f"{token}: {count}")

    evaluate_code(dataset)
