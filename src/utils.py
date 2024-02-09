"""Some utility functions"""

import time
import torch

from transformers import (
    LlamaTokenizerFast,
    AutoModelForCausalLM,
)

### Always import device to register correct backend
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# README: you should adjust based on your hardware
# NVIDIA GPUs Ampere uarch and after support BF16 (better precision than IEEE FP16)
# M2 Apple Silicon and after also support BF16 (CPU and GPU)
# Don't attempt to use FP16 on CPU, as it's not supported for GEMM
if device.type == "cuda":
    dtype = torch.bfloat16
elif device.type == "mps":
    dtype = torch.float16
else:
    # default to FP32 on CPU, because PyTorch doesn't support HGEMM on any CPU architecture
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


def load_model(model_path, tokenizer_path):
    """Load the tokenizer and model"""
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)

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
