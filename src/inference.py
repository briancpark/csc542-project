"""Here lies the inference code for the model"""
from datasets import load_dataset
from src.utils import device, load_model, torch_timer


def inference(model_path, tokenizer_path, prompt):
    """Run inference on the model"""
    tokenizer, model = load_model(model_path, tokenizer_path)

    # generate text
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # TODO: (bcp) Add support for all these parameters
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


def dataset_inference(model_path, tokenizer_path, dataset_name):
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
