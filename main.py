"""Main interface to run inference or fine-tuning"""

import argparse
import torch
from src.inference import inference, dataset_inference
from src.training import finetuning


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", type=str, default="hf-internal-testing/llama-tokenizer"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="JackFram/llama-160m",  # TinyLlama/TinyLlama-1.1B-Chat-v0.1
    )
    parser.add_argument("--temperature", "-t", type=float, default=0.0)
    parser.add_argument("--gamma", "-g", type=int, default=4)
    parser.add_argument("--n-tokens-to-generate", "-N", type=int, default=150)
    parser.add_argument("--dataset", type=str, default="openai_humaneval")
    parser.add_argument("--mode", type=str, default="sps")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Question: Write the fibonacci sequence in Python. \nAnswer: def fibonacci(n):",
    )
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--test-dataset", action="store_true")
    parser.add_argument("--finetuning", action="store_true")
    args = parser.parse_args()

    if args.inference and not args.test_dataset:
        # Disable Autograd when running inference
        with torch.no_grad():
            inference(args.model, args.tokenizer, args.prompt)
    elif args.inference and args.test_dataset:
        # Disable Autograd when running inference
        with torch.no_grad():
            dataset_inference(args.model, args.tokenizer, args.dataset)
    elif args.finetuning:
        finetuning(args.model, args.tokenizer, args.dataset)
    else:
        raise ValueError("Invalid mode.")
