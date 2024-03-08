"""Main interface to run inference or fine-tuning"""

import os
import argparse
import torch
from src.inference import inference, dataset_inference
from src.training import finetuning
from src.eda import eda


if __name__ == "__main__":
    # Set any environment variables and PyTorch performance settings
    os.environ["OMP_NUM_THREADS"] = f"{os.cpu_count()}"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", type=str, default="hf-internal-testing/llama-tokenizer"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        # default="codellama/CodeLlama-7b-hf",
    )
    parser.add_argument("--temperature", "-t", type=float, default=0.0)
    parser.add_argument("--gamma", "-g", type=int, default=4)
    parser.add_argument("--n-tokens-to-generate", "-N", type=int, default=150)
    parser.add_argument("--dataset", type=str, default="openai_humaneval")
    parser.add_argument("--test-dataset", type=str, default="openai_humaneval")
    parser.add_argument(
        "--train-dataset", type=str, default="iamtarun/code_instructions_120k_alpaca"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Question: Write the fibonacci sequence in Python. \nAnswer: def fibonacci(n):",
    )
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--inference-evaluate", action="store_true")
    parser.add_argument("--finetuning", action="store_true")
    parser.add_argument("--eda", action="store_true")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layers", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lora-checkpoint-path", type=str)
    args = parser.parse_args()

    # Run inference on a single prompt
    if args.inference:
        # Disable Autograd when running inference
        with torch.no_grad():
            inference(
                args.model,
                args.tokenizer,
                args.prompt,
                lora_checkpoint_path=args.lora_checkpoint_path,
            )
    # Evaluate inference over the whole dataset
    elif args.inference_evaluate:
        # Disable Autograd when running inference
        with torch.no_grad():
            dataset_inference(
                args.model,
                args.tokenizer,
                args.test_dataset,
                lora_checkpoint_path=args.lora_checkpoint_path,
            )
    elif args.finetuning:
        finetuning(
            args.model,
            args.tokenizer,
            args.train_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            rank=args.rank,
            alpha=args.alpha,
            layers=args.layers,
            dropout=args.dropout,
        )
    elif args.eda:
        # TODO: (bcp) We need to do EDA on the training dataset as well
        eda(args.model, args.tokenizer, args.test_dataset)
    else:
        raise ValueError("Invalid mode.")
