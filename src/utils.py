"""Some utility functions"""

import time
import torch

from torch.nn import functional as F
from transformers import (
    LlamaTokenizerFast,
    AutoModelForCausalLM,
)
from src.lora import LLaMAModelWithLoRA

### Always import device to register correct backend
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

"""
README: You should adjust based on your hardware
NVIDIA GPUs Ampere uarch and after support BF16 (better precision than IEEE FP16)
M2 Apple Silicon and after also support BF16 (CPU and GPU)
Don't attempt to use FP16 on CPU, as it's not supported for GEMM
"""
if device.type == "cuda":
    # detect if GPU is capable of BF16
    if torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
elif device.type == "mps":
    dtype = torch.float16
else:
    # Default to FP32 on CPU, because PyTorch doesn't support HGEMM on any CPU architecture
    dtype = torch.float32


def touch():
    """Synchronization primitives for the respective backends when timing routines"""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def torch_timer():
    """Timer for the respective backends"""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def load_model(
    model_path, tokenizer_path, lora=False, rank=4, lora_checkpoint_path=None
):
    """Load the tokenizer and model"""
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
    # TODO: REMOVE THIS AFTER DONE
    lora_checkpoint_path = "models/codellama_2.pt"

    if lora:
        model = LLaMAModelWithLoRA(
            model_path,
            rank=rank,
            # load_in_4bit=True,
            # load_in_8bit=True,
            # bnb_4bit_compute_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.float32,
            device_map=device,
        )

        if lora_checkpoint_path:
            # Reserialize the model, to make it platform agnostic
            model.load_state_dict(torch.load(lora_checkpoint_path, map_location="cpu"))
            model.to(device)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # load_in_4bit=True,
            # load_in_8bit=True,
            # bnb_4bit_compute_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            torch_dtype=dtype,
            device_map=device,
        )
    model.eval()

    return tokenizer, model


def sample(p, determinsitic=False):
    """Sample logits from a distribution or take the argmax"""
    if determinsitic:
        return torch.multinomial(p, 1)
    return torch.argmax(p).unsqueeze(0).unsqueeze(0)


def norm_logits(logits, temperature, eps=1e-10):
    """Normalize the logits"""
    logits = logits / (temperature + eps)
    logits = F.softmax(logits, dim=1)
    return logits


def allocated_memory():
    """Print the allocated memory in GB"""
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    elif device.type == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    else:
        return float("nan")
