"""Here outlines the training (fine-tuning) process for the model"""

import os
import json
import torch
from datasets import load_dataset
from torch import nn, LongTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import device, load_model, allocated_memory, torch_timer
from src.inference import dataset_inference


def finetuning(
    model_path,
    tokenizer_path,
    dataset_name,
    epochs=1,
    batch_size=32,
    rank=4,
    alpha=1.0,
    layers=4,
    dropout=0.0,
    lr=1e-4,
    instruction_prompt=None,
):
    """Training loop to fine-tune the model"""
    tokenizer, model = load_model(
        model_path,
        tokenizer_path,
        lora=True,
        rank=rank,
        layers=layers,
        dropout=dropout,
        alpha=alpha,
    )

    models_dir = os.path.join(os.environ["HF_HOME"], "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    display_model_name = model_path.split("/")[-1]
    losses = []

    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "iamtarun/python_code_instructions_18k_alpaca":
        dataset = load_dataset(
            "iamtarun/python_code_instructions_18k_alpaca", split="train"
        )

        def combine_columns(example):
            return {
                "prompts": '"""' + example["instruction"] + '"""\n' + example["output"]
            }

        dataset = dataset.map(combine_columns)

        def preprocess_function(examples):
            return tokenizer(
                examples["prompts"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

        dataset = dataset.map(
            preprocess_function, batched=True, remove_columns=dataset.column_names
        )
    else:
        raise ValueError("Invalid dataset name.")

    def collate_fn(batch):
        input_ids = [LongTensor(item["input_ids"]) for item in batch]
        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
        return {"input_ids": input_ids}

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=os.cpu_count(),
        collate_fn=collate_fn,
    )

    learning_rate_sci = format(lr, "e")  # Convert learning rate to scientific notation

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    pbar_epochs = tqdm(range(epochs), desc="Epochs")

    tik = torch_timer()

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

            losses.append(loss.item())
            pbar_epochs.set_description(f"Epoch {epoch}")
            pbar_batches.set_postfix(
                {"Loss": loss.item(), "Memory (GB)": allocated_memory()}
            )

        # checkpoint model at every epoch
        model_chk_path = (
            f"{models_dir}/codellama_{display_model_name}_r{rank}_a{alpha}_"
            f"l{layers}_d{dropout}_b{batch_size}_e{epoch}_lr{learning_rate_sci}.pt"
        )
        torch.save(model.state_dict(), model_chk_path)

    backprop_mem_consumed = allocated_memory()

    tok = torch_timer()
    model_chk_base = (
        f"codellama_{display_model_name}_r{rank}_a{alpha}_"
        f"l{layers}_d{dropout}_b{batch_size}_e{epochs}_lr{learning_rate_sci}_final"
    )
    model_chk_path = f"{models_dir}/{model_chk_base}.pt"
    torch.save(model.state_dict(), model_chk_path)

    # Run inference over the test dataset and log the results
    torch.cuda.empty_cache()

    model.eval()
    accuracy, execption_cnt = dataset_inference(
        model_path,
        tokenizer_path,
        "openai_humaneval",
        lora_checkpoint_path=model_chk_path,
        model=model,
        tokenizer=tokenizer,
        instruction_prompt=instruction_prompt,
    )

    results = {
        "accuracy": accuracy,
        "exception_cnt": execption_cnt,
        "loss": loss.item(),
        "params": model.trainable_params,
        "lora_params": model.lora_trainable_params,
        "memory": backprop_mem_consumed,
        "training_time": tok - tik,
        "losses": losses,
    }

    with open(f"logs/{model_chk_base}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # Taking extra precation when running with multi-gpu
    torch.cuda.empty_cache()

    return accuracy
