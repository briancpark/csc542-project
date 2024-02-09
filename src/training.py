"""Here outlines the training (fine-tuning) process for the model"""

from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from tqdm import tqdm
from src.utils import device, load_model
from src.lora import LLaMAModelWithLoRA


def finetuning(model_path, tokenizer_path, dataset_name):
    """Training loop to fine-tune the model"""
    tokenizer, model = load_model(model_path, tokenizer_path)

    if dataset_name == "openai_humaneval":
        dataset = load_dataset("openai_humaneval")
        code_prompt = dataset["test"]["prompt"]
        code_solution = dataset["test"]["completion"]
        documents = zip(code_prompt, code_solution)
    else:
        raise ValueError("Invalid dataset name.")

    lora_model = LLaMAModelWithLoRA.from_pretrained(model)

    for prompt, solution in documents:
        inputs = tokenizer(prompt + solution, return_tensors="pt").input_ids.to(device)

        labels = input_ids_tensor[
            ..., 1:
        ].contiguous()  # Shift input_ids to the left for labels
        input_ids_tensor = input_ids_tensor[
            ..., :-1
        ].contiguous()  # Remove the last token to match labels' size

        # Create a dataset and dataloader
        dataset = TensorDataset(input_ids_tensor, labels)
        dataloader = DataLoader(
            dataset, batch_size=8, shuffle=True
        )  # Adjust batch size as needed

        # Prepare optimizer
        optimizer = AdamW(lora_model.parameters(), lr=5e-5)  # Adjust learning rate as needed

        lora_model.train()  # Set model to training mode
        # TODO: (bcp) Figure out how to train with LoRA
        # Training loop
        epochs = 3  # Adjust number of epochs as needed
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                batch_input_ids, batch_labels = batch
                optimizer.zero_grad()  # Clear previous gradients

                outputs = lora_model(batch_input_ids, labels=batch_labels)
                loss = (
                    outputs.loss
                )  # Assuming you're using a model that returns a loss when labels are provided

                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model's parameters

                epoch_loss += loss.item() * batch_input_ids.size(
                    0
                )  # Aggregate the loss

            # Calculate the average loss for the epoch
            epoch_loss /= len(dataset)
            print(f"Epoch {epoch + 1} finished with average loss: {epoch_loss:.4f}")

        # After training, you might want to save your model
        # model.save_pretrained("your_model_directory")
