"""Here lies the inference code for the model"""

import torch
from tqdm import tqdm
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
    print(tokenizer.decode(output[0], skip_special_tokens=False))
    print(f"Time taken: {tok - tik:.3f} seconds")
    print(f"Tok/s: {output.shape[1] / (tok - tik):.3f}")


def dataset_inference(
    model_path, tokenizer_path, dataset_name, lora_checkpoint_path=None
):
    """Run inference on the model over a dataset"""
    lora_checkpoint_path = "models/codellama_1.pt"
    tokenizer, model = load_model(
        model_path, tokenizer_path, lora=True, lora_checkpoint_path=lora_checkpoint_path
    )

    if dataset_name == "openai_humaneval":
        dataset = load_dataset("openai_humaneval")
        examples = dataset["test"]
    else:
        raise ValueError("Invalid dataset name.")

    evaluate_code(examples, model=model, tokenizer=tokenizer)


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


def execute(code_solution, entry_point, test_function):
    """Dynamically execute the code"""
    rename_function = "candidate" + " = " + entry_point + "\n"

    code = code_solution + rename_function + test_function
    # WARNING: Using exec
    # pylint: disable-next=exec-used
    exec(code)


def evaluate_code(dataset, model=None, tokenizer=None):
    """Evaluate the code by running it"""
    instruction_prompt = """Question: Complete the following Python code. \nAnswer: """
    passed = 0
    exception_cnt = {}

    for example in tqdm(dataset):
        prompt = example["prompt"]
        test_function = example["test"]
        entry_point = example["entry_point"]

        if model and tokenizer:
            instruction_prompt_ids = tokenizer(
                instruction_prompt, return_tensors="pt"
            ).input_ids.to(device)
            instruction_prompt_idx = instruction_prompt_ids.shape[1] + 1

            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            inputs = torch.cat((instruction_prompt_ids, inputs), dim=-1)

            # TODO: might want to incorporate standard HF generate
            tik = torch_timer()
            output = autoregressive_sampling(
                inputs,
                model,
                N=250,
                temperature=1.0,
            )
            tok = torch_timer()
            output = output[:, instruction_prompt_idx:]

            solution = tokenizer.decode(output[0], skip_special_tokens=False)
            tqdm.write(f"Time taken: {tok - tik:.3f} seconds")
            tqdm.write(f"Tok/s: {output.shape[1] / (tok - tik):.3f}")

            code_solution = (
                prompt + solution[instruction_prompt_idx + 1 + len(entry_point) :]
            )
        else:
            code_solution = prompt + example["canonical_solution"]

        try:
            execute(code_solution, entry_point, test_function)
            passed += 1
        # pylint: disable=broad-exception-caught
        except Exception as e:
            exception_type = type(e).__name__
            exception_cnt[exception_type] = exception_cnt.get(exception_type, 0) + 1
            continue

    print(
        f"Accuracy: {passed / len(dataset) * 100}%; Passed: {passed} out of {len(dataset)}"
    )
    print(f"Exceptions: {exception_cnt}")
