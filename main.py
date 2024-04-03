"""Main interface to run inference or fine-tuning"""

import os
import argparse
import torch
from src.inference import inference, dataset_inference
from src.training import finetuning
from src.eda import eda
from src.tune import hpo_tune

if __name__ == "__main__":
    # Set any environment variables and PyTorch performance settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    os.environ["OMP_NUM_THREADS"] = f"{os.cpu_count()}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", type=str, default="hf-internal-testing/llama-tokenizer"
    )
    parser.add_argument(
        "--model",
        type=str,
        # default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
        # default="codellama/CodeLlama-7b-hf",
    )
    parser.add_argument("--temperature", "-t", type=float, default=0.0)
    parser.add_argument("--n-tokens-to-generate", "-N", type=int, default=500)
    parser.add_argument("--dataset", type=str, default="openai_humaneval")
    parser.add_argument("--test-dataset", type=str, default="openai_humaneval")
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="iamtarun/python_code_instructions_18k_alpaca",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Question: Write the fibonacci sequence in Python. \nAnswer: def fibonacci(n):",
    )
    parser.add_argument(
        "--instruction-prompt",
        "-ip",
        type=str,
        default="Please complete the following Python code without providing any \
                          additional tasks such as testing or explanations\n",
    )
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--inference-evaluate", action="store_true")
    parser.add_argument("--finetuning", action="store_true")
    parser.add_argument("--hyperparameter-tune", action="store_true")
    parser.add_argument("--eda", action="store_true")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-checkpoint-path", type=str)
    parser.add_argument("--debug", "-d", action="store_true")
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
                temperature=args.temperature,
                N=args.n_tokens_to_generate,
                lora_checkpoint_path=args.lora_checkpoint_path,
                instruction_prompt=args.instruction_prompt,
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
            lr=args.lr,
        )
    elif args.hyperparameter_tune:
        hpo_tune(args.model, args.tokenizer, args.train_dataset)
    elif args.eda:
        eda(args.model, args.tokenizer, args.train_dataset, training=True)
        eda(args.model, args.tokenizer, args.test_dataset, training=False)
    else:
        raise ValueError("Invalid mode.")
