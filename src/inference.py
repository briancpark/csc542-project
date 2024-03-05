"""Here lies the inference code for the model"""

import torch
from datasets import load_dataset
from src.utils import device, load_model, torch_timer, sample, norm_logits


def inference(model_path, tokenizer_path, prompt, lora_checkpoint_path=None):
    """Run inference on the model"""
    tokenizer, model = load_model(
        model_path, tokenizer_path, lora=True, lora_checkpoint_path=lora_checkpoint_path
    )
    lora_checkpoint_path = "models/codellama_0.pt"
    # generate text
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # TODO: (bcp) Add support for all these parameters
    if lora_checkpoint_path:
        tik = torch_timer()
        output = autoregressive_sampling(
            inputs,
            model,
            N=150,
            temperature=0.0,
        )
        tok = torch_timer()
    else:
        tik = torch_timer()
        output = model.generate(
            inputs,
            do_sample=False,
            temperature=0.0,
            max_length=150,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )
        tok = torch_timer()
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print(f"Time taken: {tok - tik:.3f} seconds")
    print(f"Tok/s: {output.shape[1] / (tok - tik):.3f}")


def dataset_inference(
    model_path, tokenizer_path, dataset_name, lora_checkpoint_path=None
):
    """Run inference on the model over a dataset"""
    tokenizer, model = load_model(model_path, tokenizer_path)

    if dataset_name == "openai_humaneval":
        dataset = load_dataset("openai_humaneval")
        documents = dataset["test"]["prompt"]
    else:
        raise ValueError("Invalid dataset name.")

    for prompt in documents:
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        tik = torch_timer()
        output = model.generate(
            inputs,
            do_sample=False,
            temperature=0.0,
            max_length=150,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )
        tok = torch_timer()
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print(f"Time taken: {tok - tik:.3f} seconds")
        print(f"Tok/s: {output.shape[1] / (tok - tik):.3f}")


def autoregressive_sampling(
    input_ids,
    model,
    N,
    temperature=1.0,
):
    """Autoregressive sampling from the model in inference mode"""
    n = input_ids.shape[1]
    T = input_ids.shape[1] + N

    while n < T:
        outputs = model(input_ids)
        logits = outputs.logits[::, -1, :]
        last_p = norm_logits(logits[-1:, :], temperature)
        next_token_id = sample(last_p)
        input_ids = torch.cat((input_ids, next_token_id), dim=-1)
        n += 1

    return input_ids
