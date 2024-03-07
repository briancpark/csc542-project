"""LoRA model implementation for LLaMA architecture"""

import torch
from torch import nn
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM


class LoRALinear(nn.Module):
    """LoRA (Low-Rank Adaptation) for Linear Layers"""

    def __init__(self, feature_shape, rank=1, alpha=1, dropout=0.0, device="cpu"):
        super().__init__()

        self.A = nn.Parameter(
            nn.init.normal_(torch.empty(rank, feature_shape[1]), mean=0, std=1)
        ).to(device)
        self.B = nn.Parameter(torch.zeros((feature_shape[0], rank))).to(device)

        self.scale = alpha / rank
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, W):
        """Forward pass for LoRA"""
        if self.dropout:
            return self.dropout(W) + (self.B @ self.A) * self.scale
        return W + (self.B @ self.A) * self.scale


class LLaMAModelWithLoRA(nn.Module):
    """LLaMA model enhanced with LoRA (Low-Rank Adaptation)
    More about the implementation is included in the paper:
    https://arxiv.org/pdf/2106.09685.pdf

    Note that this is mostly LLaMA architecture specific
    """

    def __init__(self, model_name_or_path, **kwargs):
        super(LLaMAModelWithLoRA, self).__init__()

        # pop off the rank parameter
        rank = kwargs.pop("rank", 4)
        alpha = kwargs.pop("alpha", 1.0)
        lora_layers = kwargs.pop("layers", 4)
        dropout = kwargs.pop("dropout", 0.0)

        device = kwargs["device_map"]
        dtype = kwargs["dtype"]

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **kwargs
        )
        self.lora_rank = rank
        self.lora_adapters = nn.ModuleList()

        self.num_layers = self.llama_model.config.num_hidden_layers
        self.hidden_size = self.llama_model.config.hidden_size

        if lora_layers == -1:
            lora_layers = 0
        else:
            lora_layers = self.num_layers - lora_layers

        self.trainable_params = sum(
            p.numel() for p in self.llama_model.parameters() if p.requires_grad
        )

        # print(self.num_layers, self.hidden_size)
        # # print dimensions of the model
        # print("dim", self.llama_model.model.layers[0].self_attn.q_proj.weight.shape)
        # print()
        # print(self.llama_model)

        # Assuming the model's transformer layer is accessible like this
        for i in range(self.num_layers):
            layer = self.llama_model.model.layers[i]

            for param in layer.parameters():
                param.requires_grad = False

            if i > lora_layers:
                # According to the paper, they only enable LoRA for q_proj and v_proj
                # print(layer)
                q_shape = layer.self_attn.q_proj.weight.shape
                layer.self_attn.q_proj = parametrize.register_parametrization(
                    layer.self_attn.q_proj,
                    "weight",
                    LoRALinear(q_shape, rank, alpha, dropout, device),
                )

                v_shape = layer.self_attn.v_proj.weight.shape
                layer.self_attn.v_proj = parametrize.register_parametrization(
                    layer.self_attn.v_proj,
                    "weight",
                    LoRALinear(v_shape, rank, alpha, dropout, device),
                )

        # now count up the number of trainable parameters
        self.lora_trainable_params = sum(
            p.numel() for p in self.llama_model.parameters() if p.requires_grad
        )

    def forward(self, input_ids, labels=None):
        """
        Forward pass that integrates LoRA adjustments.
        This requires custom handling based on the internal structure of the LLaMA model.
        """

        output = self.llama_model(
            input_ids=input_ids,
            labels=labels,
        )

        return output

    def merge(self):
        """Merge the LoRA weights back into the model"""

        for i in range(self.num_layers):
            layer = self.llama_model.model.layers[i]

            # Multiply W with (B @ A) and store it back in W
            layer.self_attn.q_proj.weight = (
                layer.self_attn.q_proj.weight
                + (layer.self_attn.q_proj.B @ layer.self_attn.q_proj.A)
                * layer.self_attn.q_proj.scale
            )
            layer.self_attn.v_proj.weight = (
                layer.self_attn.v_proj.weight
                + (layer.self_attn.v_proj.B @ layer.self_attn.v_proj.A)
                * layer.self_attn.v_proj.scale
            )
