"""Here lies the inference code for the model"""

import signal
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
            tokenizer,
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
    if lora_checkpoint_path:
        tokenizer, model = load_model(
            model_path,
            tokenizer_path,
            lora=True,
            rank=8,
            layers=-1,
            alpha=1.0,
            dropout=0.1,
            lora_checkpoint_path=lora_checkpoint_path,
        )
    else:
        tokenizer, model = load_model(
            model_path,
            tokenizer_path,
        )

    if dataset_name == "openai_humaneval":
        dataset = load_dataset("openai_humaneval")
        examples = dataset["test"]
    else:
        raise ValueError("Invalid dataset name.")

    return evaluate_code(examples, model=model, tokenizer=tokenizer)


def autoregressive_sampling(
    input_ids,
    model,
    tokenizer,
    N,
    temperature=1.0,
):
    """Autoregressive sampling from the model in inference mode"""
    n = input_ids.shape[1]
    T = input_ids.shape[1] + N

    while n < T:
        outputs = model(input_ids)
        logits = outputs.logits[::, -1, :]
        # Apply repetition penalty
        p = norm_logits(logits[-1:, :], temperature)
        next_token_id = sample(p, deterministic=True)
        # Add the generated token to the set of generated tokens
        input_ids = torch.cat((input_ids, next_token_id), dim=-1)
        n += 1

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    return input_ids


def execute(code_solution, entry_point, test_function):
    """Dynamically execute the code"""
    main_fn = f"""candidate = {entry_point}\ncheck(candidate)\n"""

    code = code_solution + "\n" + test_function + "\n" + main_fn
    print(code)

    # Sometimes code will require input or run in infinite loops, so just time it out

    def handler(signum, frame):
        raise TimeoutError("Execution timed out")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60)

    try:
        # WARNING: Using exec
        # pylint: disable-next=exec-used
        exec(code)
    finally:
        # Disable the alarm
        signal.alarm(0)


def evaluate_code(dataset, model=None, tokenizer=None):
    """Evaluate the code by running it"""
    instruction_prompt = (
        """Complete the following Python code without any tests or explanation\n"""
    )
    passed = 0
    exception_cnt = {}

    for example in tqdm(dataset):
        prompt = example["prompt"] + "\n    "
        test_function = example["test"]
        entry_point = example["entry_point"]

        if model and tokenizer:
            instruction_prompt_ids = tokenizer(
                instruction_prompt, return_tensors="pt"
            ).input_ids.to(device)

            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            inputs = torch.cat((instruction_prompt_ids, inputs), dim=-1)

            inputs_idx = inputs.shape[1]

            # TODO: might want to incorporate standard HF generate
            tik = torch_timer()
            output = autoregressive_sampling(
                inputs,
                model,
                tokenizer,
                N=inputs_idx + 250,
                temperature=0.0,
            )
            tok = torch_timer()

            solution = tokenizer.decode(
                output[0, inputs_idx:],
                skip_special_tokens=True,
                # clean_up_tokenization_spaces=False,
            )
            tqdm.write(f"Time taken: {tok - tik:.3f} seconds")
            tqdm.write(f"Tok/s: {output.shape[1] / (tok - tik):.3f}")

            code_solution = prompt + solution

            # remove </s>
            code_solution = code_solution.replace("</s>", "")
            # README: This is for CodeLLaMA; but we need to just take extra
            # precaution to remove the last line
            # It's very hacky, but I cannot find any other systematic way around it.
            # Trim anything after if __name__ == "__main__":
            code_solution = code_solution.split('if __name__ == "__main__":')[0]
        else:
            code_solution = prompt + example["canonical_solution"]

        try:
            execute(code_solution, entry_point, test_function)
            passed += 1
            print("Passed!")
        # pylint: disable=broad-exception-caught
        except Exception as e:
            exception_type = type(e).__name__
            print(e)
            exception_cnt[exception_type] = exception_cnt.get(exception_type, 0) + 1
            print(passed, exception_cnt)
            continue
        print(passed, exception_cnt)
    print(
        f"Accuracy: {passed / len(dataset) * 100}%; Passed: {passed} out of {len(dataset)}"
    )
    print(f"Exceptions: {exception_cnt}")
    return passed / len(dataset), exception_cnt
