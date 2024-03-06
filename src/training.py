"""Here outlines the training (fine-tuning) process for the model"""

import os
import json
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.utils import device, load_model, allocated_memory
from src.inference import dataset_inference


class HumanEvalHFDataSet(Dataset):
    """Human Eval dataset for Hugging Face"""

    def __init__(self, tokenizer, hf_dataset, block_size=512):
        self.examples = []

        # Iterate through the dataset and prepare the inputs and labels
        for prompt, solution in zip(
            hf_dataset["prompt"], hf_dataset["canonical_solution"]
        ):
            # Concatenate prompt and solution for the full context
            full_text = prompt + solution
            self.examples.append(
                tokenizer(
                    full_text,
                    truncation=True,
                    padding="max_length",  # Add this line
                    max_length=block_size,
                    return_tensors="pt",
                )
            )

        self.block_size = block_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Here, each item is a dictionary of tensors
        return {key: val.squeeze(0) for key, val in self.examples[idx].items()}


class AlpacaHFDataSet(Dataset):
    """Alpaca dataset for Hugging Face"""

    def __init__(self, tokenizer, hf_dataset, block_size=600):
        self.examples = []

        # Iterate through the dataset and prepare the inputs and labels
        for prompt in hf_dataset["prompt"]:
            self.examples.append(
                tokenizer(
                    prompt,
                    truncation=True,
                    padding="max_length",  # Add this line
                    max_length=block_size,
                    return_tensors="pt",
                )
            )

        self.block_size = block_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Here, each item is a dictionary of tensors
        return {key: val.squeeze(0) for key, val in self.examples[idx].items()}


def finetuning(
    model_path,
    tokenizer_path,
    dataset_name,
    epochs=1,
    batch_size=32,
    rank=4,
    alpha=1.0,
    layers=4,
):
    """Training loop to fine-tune the model"""
    tokenizer, model = load_model(
        model_path, tokenizer_path, lora=True, rank=rank, layers=layers, alpha=alpha
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    display_model_name = model_path.split("/")[-1]

    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "openai_humaneval":
        dataset = load_dataset("openai_humaneval", split="test")
        dataset = HumanEvalHFDataSet(tokenizer, dataset)
    elif dataset_name == "iamtarun/code_instructions_120k_alpaca":
        dataset = load_dataset("iamtarun/code_instructions_120k_alpaca", split="train")
        dataset = AlpacaHFDataSet(tokenizer, dataset)
    else:
        raise ValueError("Invalid dataset name.")

    data_loader = DataLoader(dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    pbar_epochs = tqdm(range(epochs), desc="Epochs")
    for epoch in pbar_epochs:
        pbar_batches = tqdm(data_loader, desc="Batches", leave=False)
        for batch in pbar_batches:
            # For LM, input and labels are usually the same
            inputs = batch["input_ids"].to(device)
            labels = batch["input_ids"].to(device)

            outputs = model(inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar_epochs.set_description(f"Epoch {epoch}")
            pbar_batches.set_postfix(
                {"Loss": loss.item(), "Memory (GB)": allocated_memory()}
            )
        # checkpoint model at every epoch
        torch.save(
            model.state_dict(),
            f"models/codellama_{display_model_name}_\
                {rank}_\
                {alpha}_\
                {layers}_\
                {batch_size}_\
                {epoch}.pt",
        )

    backprop_mem_consumed = allocated_memory()
    torch.save(
        model.state_dict(),
        f"models/codellama_{display_model_name}_\
            {rank}_\
            {alpha}_\
            {layers}_\
            {batch_size}_\
            {epochs}_\
            final.pt",
    )

    # Run inference over the test dataset and log the results

    model.eval()
    accuracy, execption_cnt = dataset_inference(
        model_path,
        tokenizer_path,
        "openai_humaneval",
        lora_checkpoint_path=f"models/codellama_{display_model_name}_\
            {rank}_\
            {alpha}_\
            {layers}_\
            {batch_size}_\
            {epochs}_\
            final.pt",
    )

    results = {
        "accuracy": accuracy,
        "exception_cnt": execption_cnt,
        "loss": loss.item(),
        "params": model.trainable_params,
        "lora_params": model.lora_trainable_params,
        "total_params": sum(p.numel() for p in model.parameters()),
        "memory": backprop_mem_consumed,
    }

    with open(
        f"logs/codellama_{display_model_name}_\
        {rank}_\
        {alpha}_\
        {layers}_\
        {batch_size}_\
        {epochs}_\
        final.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=4)
