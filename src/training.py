"""Here outlines the training (fine-tuning) process for the model"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.utils import device, load_model


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


def finetuning(model_path, tokenizer_path, dataset_name, epochs=10):
    """Training loop to fine-tune the model"""
    tokenizer, model = load_model(model_path, tokenizer_path, lora=True)

    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "openai_humaneval":
        dataset = load_dataset("openai_humaneval", split="test")
    else:
        raise ValueError("Invalid dataset name.")

    dataset = HumanEvalHFDataSet(tokenizer, dataset)

    data_loader = DataLoader(dataset, batch_size=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in tqdm(range(epochs)):
        for batch in data_loader:
            # For LM, input and labels are usually the same
            inputs = batch["input_ids"].to(device)
            labels = batch["input_ids"].to(device)

            outputs = model(inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")
