#!/usr/bin/env python3
"""
Comprehensive nanoGPT training and analysis script with entropy tracking.
Supports training on OpenWebText, random tokens, and constant tokens.
Includes detailed entropy analysis of model components and KV cache.
"""

import os
import time
import math
import json
from contextlib import nullcontext
from dataclasses import dataclass
import inspect

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================================================================================================
# MODEL DEFINITION (Mostly verbatim from original nanoGPT model.py)
# ================================================================================================


# === VERBATIM FROM model.py ===
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# === MODIFIED FROM model.py (added KV cache support) ===
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x, return_kv_cache=False):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        # MODIFICATION: return KV cache if requested
        if return_kv_cache:
            return y, (k, v)
        return y


# === VERBATIM FROM model.py ===
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# === MODIFIED FROM model.py (added KV cache support) ===
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, return_kv_cache=False):
        # MODIFICATION: support KV cache return
        if return_kv_cache:
            attn_out, kv_cache = self.attn(self.ln_1(x), return_kv_cache=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, kv_cache
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


# === VERBATIM FROM model.py ===
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


# === HEAVILY MODIFIED FROM model.py (removed from_pretrained, added KV cache support, removed weight tying warning comments) ===
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print(f"Initialized model with {self.get_num_params()/1e6:.2f}M parameters")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_kv_cache=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # MODIFICATION: collect KV caches if requested
        kv_caches = []
        for block in self.transformer.h:
            if return_kv_cache:
                x, kv_cache = block(x, return_kv_cache=True)
                kv_caches.append(kv_cache)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        if return_kv_cache:
            return logits, loss, kv_caches
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # === SIMILAR TO model.py configure_optimizers ===
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"Optimizer setup: {len(decay_params):,} decay tensors ({num_decay_params:,} params), "
            f"{len(nodecay_params):,} no-decay tensors ({num_nodecay_params:,} params)"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"Using {'fused' if use_fused else 'standard'} AdamW optimizer")

        return optimizer

    # === SIMILAR TO model.py estimate_mfu ===
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    # === MODIFIED FROM model.py generate (added KV cache support) ===
    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, return_kv_cache=False
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        all_kv_caches = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            if return_kv_cache:
                logits, _, kv_caches = self(idx_cond, return_kv_cache=True)
                all_kv_caches.append(kv_caches)
            else:
                logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        if return_kv_cache:
            return idx, all_kv_caches
        return idx


# ================================================================================================
# ENTROPY TRACKING UTILITIES (New implementation)
# ================================================================================================


class EntropyTracker:
    """Tracks entropy of model parameters and activations during training"""

    def __init__(self, num_bins=100):
        self.num_bins = num_bins
        self.entropy_history = {}
        self.iteration_history = []

    def calculate_entropy(self, tensor, bins=None):
        """Calculate entropy of a tensor using histogram binning"""
        if tensor.numel() == 0:
            return 0.0

        # Flatten tensor and convert to numpy
        flat_tensor = tensor.detach().cpu().flatten().numpy()

        # Remove any NaN or inf values
        flat_tensor = flat_tensor[np.isfinite(flat_tensor)]
        if len(flat_tensor) == 0:
            return 0.0

        # Calculate histogram
        bins = bins or self.num_bins
        hist, _ = np.histogram(flat_tensor, bins=bins, density=True)

        # Normalize to get probabilities
        hist = hist + 1e-10  # Add small epsilon to avoid log(0)
        prob = hist / np.sum(hist)

        # Calculate entropy in bits
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        return entropy

    def calculate_kv_cache_entropy(self, kv_caches):
        """Calculate entropy of KV cache tensors"""
        k_entropies = []
        v_entropies = []

        for layer_cache in kv_caches:
            k, v = layer_cache
            k_entropy = self.calculate_entropy(k)
            v_entropy = self.calculate_entropy(v)
            k_entropies.append(k_entropy)
            v_entropies.append(v_entropy)

        return {
            "k_entropies": k_entropies,
            "v_entropies": v_entropies,
            "total_k_entropy": np.mean(k_entropies),
            "total_v_entropy": np.mean(v_entropies),
            "total_kv_entropy": np.mean(k_entropies + v_entropies),
        }

    def log_entropy(self, model, iteration, track_layers=False, track_components=False):
        """Log entropy of model parameters"""
        entropies = {}

        # Calculate total parameter entropy
        all_params = torch.cat([p.flatten() for p in model.parameters()])
        entropies["entropy_total"] = self.calculate_entropy(all_params)

        if track_components:
            # Track entropy by component
            entropies["entropy_embedding"] = self.calculate_entropy(
                model.transformer.wte.weight
            )
            entropies["entropy_pos_embedding"] = self.calculate_entropy(
                model.transformer.wpe.weight
            )
            entropies["entropy_lm_head"] = self.calculate_entropy(model.lm_head.weight)

            # Track layer norm entropy
            ln_params = []
            for name, param in model.named_parameters():
                if "ln_" in name or "layernorm" in name.lower():
                    ln_params.append(param.flatten())
            if ln_params:
                entropies["entropy_layernorm"] = self.calculate_entropy(
                    torch.cat(ln_params)
                )

            # Track attention entropy
            attn_params = []
            mlp_params = []
            for name, param in model.named_parameters():
                if "attn" in name:
                    attn_params.append(param.flatten())
                elif "mlp" in name:
                    mlp_params.append(param.flatten())

            if attn_params:
                entropies["entropy_attention"] = self.calculate_entropy(
                    torch.cat(attn_params)
                )
            if mlp_params:
                entropies["entropy_mlp"] = self.calculate_entropy(torch.cat(mlp_params))

        if track_layers:
            # Track entropy by layer
            for i, block in enumerate(model.transformer.h):
                layer_params = torch.cat([p.flatten() for p in block.parameters()])
                entropies[f"entropy_layer_{i}"] = self.calculate_entropy(layer_params)

        # Store in history
        for key, value in entropies.items():
            if key not in self.entropy_history:
                self.entropy_history[key] = []
            self.entropy_history[key].append(value)

        # Store iteration
        if len(self.iteration_history) == 0 or self.iteration_history[-1] != iteration:
            self.iteration_history.append(iteration)

        return entropies

    def get_latest_entropy(self):
        """Get the most recent entropy measurements"""
        latest = {}
        for key, values in self.entropy_history.items():
            if values:
                latest[key] = values[-1]
        return latest

    def _synchronize_arrays(self):
        """Ensure all entropy history arrays have the same length as iteration_history"""
        if not self.iteration_history:
            return

        target_length = len(self.iteration_history)

        # Synchronize all entropy history arrays
        for key in list(self.entropy_history.keys()):
            if isinstance(self.entropy_history[key], list):
                current_length = len(self.entropy_history[key])
                if current_length != target_length:
                    # Truncate or pad to match target length
                    if current_length > target_length:
                        self.entropy_history[key] = self.entropy_history[key][
                            :target_length
                        ]
                    else:
                        # Pad with the last value if shorter
                        if current_length > 0:
                            last_value = self.entropy_history[key][-1]
                            self.entropy_history[key].extend(
                                [last_value] * (target_length - current_length)
                            )
                        else:
                            # If completely empty, pad with zeros
                            self.entropy_history[key] = [0.0] * target_length

    def plot_entropy_evolution(self, save_path=None, show_layers=False):
        """Plot entropy evolution over training"""
        try:
            # First, synchronize all arrays to the same length
            self._synchronize_arrays()

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Parameter Entropy Evolution During Training")

            # Plot total entropy
            if (
                "entropy_total" in self.entropy_history
                and len(self.entropy_history["entropy_total"]) > 0
            ):
                entropy_data = self.entropy_history["entropy_total"]
                iter_data = self.iteration_history

                print(
                    f"   Plotting total entropy: {len(iter_data)} iterations, {len(entropy_data)} entropy points"
                )

                if (
                    len(entropy_data) > 0
                    and len(iter_data) > 0
                    and len(entropy_data) == len(iter_data)
                ):
                    axes[0, 0].plot(
                        iter_data,
                        entropy_data,
                        "b-",
                        linewidth=2,
                    )
                    axes[0, 0].set_title("Total Parameter Entropy")
                    axes[0, 0].set_xlabel("Iteration")
                    axes[0, 0].set_ylabel("Entropy (bits)")
                    axes[0, 0].grid(True)
                else:
                    print(
                        f"   Skipping total entropy plot due to length mismatch: iter={len(iter_data)}, entropy={len(entropy_data)}"
                    )

            # Plot component entropies if available
            component_keys = [
                k
                for k in self.entropy_history.keys()
                if k.startswith("entropy_")
                and k != "entropy_total"
                and not k.startswith("entropy_layer_")
            ]

            has_component_plots = False
            if component_keys:
                iter_data = self.iteration_history
                for key in component_keys:
                    if (
                        key in self.entropy_history
                        and len(self.entropy_history[key]) > 0
                    ):
                        entropy_data = self.entropy_history[key]

                        if (
                            len(entropy_data) == len(iter_data)
                            and len(entropy_data) > 0
                        ):
                            label = (
                                key.replace("entropy_", "").replace("_", " ").title()
                            )
                            axes[0, 1].plot(
                                iter_data,
                                entropy_data,
                                label=label,
                                linewidth=2,
                            )
                            has_component_plots = True
                        else:
                            print(
                                f"   Skipping {key}: length mismatch iter={len(iter_data)}, entropy={len(entropy_data)}"
                            )

            if has_component_plots:
                axes[0, 1].set_title("Component Entropy")
                axes[0, 1].set_xlabel("Iteration")
                axes[0, 1].set_ylabel("Entropy (bits)")
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "No component data\navailable",
                    ha="center",
                    va="center",
                    transform=axes[0, 1].transAxes,
                )
                axes[0, 1].set_title("Component Entropy (No Data)")

            # Plot layer entropies if available
            layer_keys = [
                k for k in self.entropy_history.keys() if k.startswith("entropy_layer_")
            ]

            has_layer_plots = False
            if layer_keys and show_layers:
                iter_data = self.iteration_history
                for key in layer_keys:
                    if (
                        key in self.entropy_history
                        and len(self.entropy_history[key]) > 0
                    ):
                        entropy_data = self.entropy_history[key]

                        if (
                            len(entropy_data) == len(iter_data)
                            and len(entropy_data) > 0
                        ):
                            layer_num = key.replace("entropy_layer_", "")
                            axes[1, 0].plot(
                                iter_data,
                                entropy_data,
                                label=f"Layer {layer_num}",
                                linewidth=1.5,
                            )
                            has_layer_plots = True
                        else:
                            print(
                                f"   Skipping {key}: length mismatch iter={len(iter_data)}, entropy={len(entropy_data)}"
                            )

            if has_layer_plots:
                axes[1, 0].set_title("Layer-wise Entropy")
                axes[1, 0].set_xlabel("Iteration")
                axes[1, 0].set_ylabel("Entropy (bits)")
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            else:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "No layer data\navailable",
                    ha="center",
                    va="center",
                    transform=axes[1, 0].transAxes,
                )
                axes[1, 0].set_title("Layer Entropy (No Data)")

            # Plot entropy distribution
            if (
                "entropy_total" in self.entropy_history
                and len(self.entropy_history["entropy_total"]) > 0
            ):
                entropy_data = self.entropy_history["entropy_total"]
                if len(entropy_data) > 0:
                    axes[1, 1].hist(
                        entropy_data,
                        bins=min(
                            20, len(set(entropy_data))
                        ),  # Adjust bins based on unique values
                        alpha=0.7,
                        color="skyblue",
                        edgecolor="black",
                    )
                    axes[1, 1].set_title("Total Entropy Distribution")
                    axes[1, 1].set_xlabel("Entropy (bits)")
                    axes[1, 1].set_ylabel("Frequency")
                    axes[1, 1].grid(True)
                else:
                    axes[1, 1].text(
                        0.5,
                        0.5,
                        "No entropy data\nfor distribution",
                        ha="center",
                        va="center",
                        transform=axes[1, 1].transAxes,
                    )
                    axes[1, 1].set_title("Entropy Distribution (No Data)")
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "No entropy data\nfor distribution",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("Entropy Distribution (No Data)")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"📊 Entropy evolution plot saved: {save_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            print(f"❌ Error creating entropy evolution plot: {e}")
            import traceback

            traceback.print_exc()
            if "fig" in locals():
                plt.close(fig)


def analyze_entropy_scaling(entropy_tracker):
    """Analyze entropy scaling properties"""
    if (
        "entropy_total" not in entropy_tracker.entropy_history
        or len(entropy_tracker.entropy_history["entropy_total"]) < 10
    ):
        return {}

    entropies = np.array(entropy_tracker.entropy_history["entropy_total"])
    iterations = np.array(entropy_tracker.iteration_history)

    # Ensure arrays have the same length
    min_len = min(len(entropies), len(iterations))
    entropies = entropies[:min_len]
    iterations = iterations[:min_len]

    # Calculate entropy change rate
    if len(entropies) > 1 and len(iterations) > 1:
        try:
            entropy_change_rate = np.diff(entropies) / np.diff(iterations)
            mean_change_rate = np.mean(entropy_change_rate)
            std_change_rate = np.std(entropy_change_rate)
        except Exception as e:
            print(f"Warning: Could not calculate entropy change rate: {e}")
            mean_change_rate = 0.0
            std_change_rate = 0.0
    else:
        mean_change_rate = 0.0
        std_change_rate = 0.0

    # Calculate entropy stability (coefficient of variation)
    entropy_cv = np.std(entropies) / (np.mean(entropies) + 1e-10)

    # Final entropy values
    final_entropy = entropies[-1] if len(entropies) > 0 else 0.0
    initial_entropy = entropies[0] if len(entropies) > 0 else 0.0

    return {
        "final_entropy": final_entropy,
        "initial_entropy": initial_entropy,
        "entropy_change": final_entropy - initial_entropy,
        "mean_change_rate": mean_change_rate,
        "std_change_rate": std_change_rate,
        "entropy_coefficient_variation": entropy_cv,
        "max_entropy": np.max(entropies) if len(entropies) > 0 else 0.0,
        "min_entropy": np.min(entropies) if len(entropies) > 0 else 0.0,
    }


# ================================================================================================
# DATA LOADING UTILITIES (Similar to original prepare.py files)
# ================================================================================================


class DataLoader:
    """Handles different types of data loading"""

    def __init__(
        self, dataset_type, vocab_size=50304, block_size=1024, data_dir="data"
    ):
        self.dataset_type = dataset_type
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.data_dir = data_dir

        print(f"📁 Preparing {dataset_type} dataset...")

        if dataset_type == "openwebtext":
            self.prepare_openwebtext()
        elif dataset_type == "random":
            self.prepare_random_data()
        elif dataset_type == "constant":
            self.prepare_constant_data()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    # === SIMILAR TO data/openwebtext/prepare.py ===
    def prepare_openwebtext(self):
        """Prepare OpenWebText dataset"""
        data_path = os.path.join(self.data_dir, "openwebtext")
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train.bin")
        val_path = os.path.join(data_path, "val.bin")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print("🌐 Downloading and tokenizing OpenWebText dataset...")

            # Load dataset
            dataset = load_dataset("openwebtext", num_proc=8)

            # Create train/val split
            split_dataset = dataset["train"].train_test_split(
                test_size=0.0005, seed=2357, shuffle=True
            )
            split_dataset["val"] = split_dataset.pop("test")

            # Tokenize
            enc = tiktoken.get_encoding("gpt2")

            def process(example):
                ids = enc.encode_ordinary(example["text"])
                ids.append(enc.eot_token)
                return {"ids": ids, "len": len(ids)}

            tokenized = split_dataset.map(
                process,
                remove_columns=["text"],
                desc="tokenizing the splits",
                num_proc=8,
            )

            # Save to binary files
            for split, dset in tokenized.items():
                arr_len = np.sum(dset["len"], dtype=np.uint64)
                filename = os.path.join(data_path, f"{split}.bin")
                dtype = np.uint16
                arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
                total_batches = 1024

                idx = 0
                for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                    batch = dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    ).with_format("numpy")
                    arr_batch = np.concatenate(batch["ids"])
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()

            print("✅ OpenWebText dataset prepared successfully!")
        else:
            print("✅ OpenWebText dataset already exists, skipping preparation")

        self.data_path = data_path

    def prepare_random_data(self):
        """Prepare random token dataset"""
        data_path = os.path.join(self.data_dir, "random")
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train.bin")
        val_path = os.path.join(data_path, "val.bin")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            # Generate random data
            train_size = 10000000  # 10M tokens
            val_size = 100000  # 100K tokens

            print(
                f"🎲 Generating random token dataset ({train_size:,} train, {val_size:,} val tokens)..."
            )
            train_data = np.random.randint(
                0, self.vocab_size, size=train_size, dtype=np.uint16
            )
            val_data = np.random.randint(
                0, self.vocab_size, size=val_size, dtype=np.uint16
            )

            train_data.tofile(train_path)
            val_data.tofile(val_path)
            print("✅ Random token dataset created!")
        else:
            print("✅ Random token dataset already exists, skipping generation")

        self.data_path = data_path

    def prepare_constant_data(self):
        """Prepare constant token dataset (all token 1)"""
        data_path = os.path.join(self.data_dir, "constant")
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train.bin")
        val_path = os.path.join(data_path, "val.bin")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            # Generate constant data (all token 1)
            train_size = 10000000  # 10M tokens
            val_size = 100000  # 100K tokens

            print(
                f"🔢 Generating constant token dataset (all token 1, {train_size:,} train, {val_size:,} val tokens)..."
            )
            train_data = np.ones(train_size, dtype=np.uint16)
            val_data = np.ones(val_size, dtype=np.uint16)

            train_data.tofile(train_path)
            val_data.tofile(val_path)
            print("✅ Constant token dataset created!")
        else:
            print("✅ Constant token dataset already exists, skipping generation")

        self.data_path = data_path

    # === SIMILAR TO get_batch from train.py ===
    def get_batch(self, split, batch_size):
        """Get a batch of data"""
        if split == "train":
            data = np.memmap(
                os.path.join(self.data_path, "train.bin"), dtype=np.uint16, mode="r"
            )
        else:
            data = np.memmap(
                os.path.join(self.data_path, "val.bin"), dtype=np.uint16, mode="r"
            )

        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )

        return x, y


# ================================================================================================
# TRAINING UTILITIES (Similar to train.py functions)
# ================================================================================================


def create_model_configs():
    """Create model configurations for different sizes"""
    configs = {
        "gpt2_half": GPTConfig(
            n_layer=8,  # Reduced from 12 to get ~62M params
            n_head=8,  # Reduced from 12
            n_embd=512,  # Reduced from 768
            block_size=1024,
            vocab_size=50304,
            dropout=0.0,
            bias=False,
        ),
        "small_full_vocab": GPTConfig(
            n_layer=6,
            n_head=6,
            n_embd=384,
            block_size=1024,
            vocab_size=50304,  # Full vocab size
            dropout=0.0,
            bias=False,
        ),
    }
    return configs


# === SIMILAR TO estimate_loss from train.py ===
@torch.no_grad()
def estimate_loss(model, data_loader, eval_iters, ctx, device):
    """Estimate loss over multiple batches"""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data_loader.get_batch(split, batch_size=8)  # Small batch for eval
            X, Y = X.to(device), Y.to(device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# === SIMILAR TO get_lr from train.py ===
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Learning rate scheduler with cosine decay"""
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def analyze_trained_model_data_only(model, data_loader, device, ctx, output_dir):
    """Analyze trained model and save data only (no plots)"""
    print(f"\n{'🔍 ANALYZING TRAINED MODEL (DATA ONLY)':=^80}")

    # Create entropy tracker for analysis
    entropy_tracker = EntropyTracker(num_bins=100)

    # Log entropy with detailed component and layer tracking
    print("📊 Computing parameter entropy analysis...")
    entropies = entropy_tracker.log_entropy(
        model, 0, track_layers=True, track_components=True
    )

    print(f"\n{'Parameter Entropy Analysis':^60}")
    print("─" * 60)
    for key, value in sorted(entropies.items()):
        display_name = key.replace("entropy_", "").replace("_", " ").title()
        print(f"{display_name:.<40} {value:>8.6f} bits")

    # Sample some text and analyze KV cache entropy
    print(f"\n{'KV Cache Entropy Analysis':^60}")
    print("─" * 60)

    enc = tiktoken.get_encoding("gpt2")

    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
        "Hello world, this is",
        "Machine learning is",
    ]

    model.eval()
    kv_cache_entropies = []

    for i, prompt in enumerate(prompts):
        print(f"\n🔤 Prompt {i+1}: '{prompt}'")

        # Encode prompt
        start_ids = enc.encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

        # Generate with KV cache tracking
        with torch.no_grad():
            with ctx:
                generated, all_kv_caches = model.generate(
                    x, 20, temperature=0.8, return_kv_cache=True
                )

        # Analyze KV cache entropy for the last generation step
        if all_kv_caches:
            last_kv_cache = all_kv_caches[-1]  # Last generation step
            kv_entropy = entropy_tracker.calculate_kv_cache_entropy(last_kv_cache)
            kv_cache_entropies.append(kv_entropy)

            print(f"   Total KV Entropy: {kv_entropy['total_kv_entropy']:8.6f} bits")
            print(f"   K Entropy:       {kv_entropy['total_k_entropy']:8.6f} bits")
            print(f"   V Entropy:       {kv_entropy['total_v_entropy']:8.6f} bits")

            # Show generated text
            generated_text = enc.decode(generated[0].tolist())
            print(f"   Generated: '{generated_text}'")

    # Calculate KV cache statistics
    if kv_cache_entropies:
        kv_analysis = {
            "prompts": prompts,
            "kv_cache_entropies": kv_cache_entropies,
            "average_kv_entropy": np.mean(
                [kv["total_kv_entropy"] for kv in kv_cache_entropies]
            ),
            "average_k_entropy": np.mean(
                [kv["total_k_entropy"] for kv in kv_cache_entropies]
            ),
            "average_v_entropy": np.mean(
                [kv["total_v_entropy"] for kv in kv_cache_entropies]
            ),
            "std_kv_entropy": np.std(
                [kv["total_kv_entropy"] for kv in kv_cache_entropies]
            ),
            "std_k_entropy": np.std(
                [kv["total_k_entropy"] for kv in kv_cache_entropies]
            ),
            "std_v_entropy": np.std(
                [kv["total_v_entropy"] for kv in kv_cache_entropies]
            ),
        }

        print(f"\n{'KV Cache Statistics':^60}")
        print("─" * 60)
        print(
            f"Average KV Entropy:     {kv_analysis['average_kv_entropy']:8.6f} ± {kv_analysis['std_kv_entropy']:6.6f} bits"
        )
        print(
            f"Average K Entropy:      {kv_analysis['average_k_entropy']:8.6f} ± {kv_analysis['std_k_entropy']:6.6f} bits"
        )
        print(
            f"Average V Entropy:      {kv_analysis['average_v_entropy']:8.6f} ± {kv_analysis['std_v_entropy']:6.6f} bits"
        )

        # Save KV cache analysis
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        with open(os.path.join(output_dir, "kv_cache_analysis.json"), "w") as f:
            json.dump(convert_numpy(kv_analysis), f, indent=2)

        print(f"💾 KV Cache analysis saved: {output_dir}/kv_cache_analysis.json")

    # Save detailed parameter analysis
    parameter_analysis = {
        "model_size": model.get_num_params(),
        "parameter_entropies": entropies,
        "entropy_scaling": analyze_entropy_scaling(entropy_tracker),
    }

    with open(os.path.join(output_dir, "parameter_analysis.json"), "w") as f:
        json.dump(parameter_analysis, f, indent=2)

    print(f"💾 Parameter analysis saved: {output_dir}/parameter_analysis.json")

    return entropies


def analyze_trained_model(model, data_loader, device, ctx, output_dir):
    """Comprehensive analysis of a trained model"""
    print(f"\n{'🔍 ANALYZING TRAINED MODEL':=^80}")

    # Create entropy tracker for analysis
    entropy_tracker = EntropyTracker(num_bins=100)

    # Log entropy with detailed component and layer tracking
    print("📊 Computing parameter entropy analysis...")
    entropies = entropy_tracker.log_entropy(
        model, 0, track_layers=True, track_components=True
    )

    print(f"\n{'Parameter Entropy Analysis':^60}")
    print("─" * 60)
    for key, value in sorted(entropies.items()):
        display_name = key.replace("entropy_", "").replace("_", " ").title()
        print(f"{display_name:.<40} {value:>8.6f} bits")

    # Sample some text and analyze KV cache entropy
    print(f"\n{'KV Cache Entropy Analysis':^60}")
    print("─" * 60)

    enc = tiktoken.get_encoding("gpt2")

    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
        "Hello world, this is",
        "Machine learning is",
    ]

    model.eval()
    kv_cache_entropies = []

    for i, prompt in enumerate(prompts):
        print(f"\n🔤 Prompt {i+1}: '{prompt}'")

        # Encode prompt
        start_ids = enc.encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

        # Generate with KV cache tracking
        with torch.no_grad():
            with ctx:
                generated, all_kv_caches = model.generate(
                    x, 20, temperature=0.8, return_kv_cache=True
                )

        # Analyze KV cache entropy for the last generation step
        if all_kv_caches:
            last_kv_cache = all_kv_caches[-1]  # Last generation step
            kv_entropy = entropy_tracker.calculate_kv_cache_entropy(last_kv_cache)
            kv_cache_entropies.append(kv_entropy)

            print(f"   Total KV Entropy: {kv_entropy['total_kv_entropy']:8.6f} bits")
            print(f"   K Entropy:       {kv_entropy['total_k_entropy']:8.6f} bits")
            print(f"   V Entropy:       {kv_entropy['total_v_entropy']:8.6f} bits")

            # Show generated text
            generated_text = enc.decode(generated[0].tolist())
            print(f"   Generated: '{generated_text}'")

    # Calculate KV cache statistics
    if kv_cache_entropies:
        kv_analysis = {
            "prompts": prompts,
            "kv_cache_entropies": kv_cache_entropies,
            "average_kv_entropy": np.mean(
                [kv["total_kv_entropy"] for kv in kv_cache_entropies]
            ),
            "average_k_entropy": np.mean(
                [kv["total_k_entropy"] for kv in kv_cache_entropies]
            ),
            "average_v_entropy": np.mean(
                [kv["total_v_entropy"] for kv in kv_cache_entropies]
            ),
            "std_kv_entropy": np.std(
                [kv["total_kv_entropy"] for kv in kv_cache_entropies]
            ),
            "std_k_entropy": np.std(
                [kv["total_k_entropy"] for kv in kv_cache_entropies]
            ),
            "std_v_entropy": np.std(
                [kv["total_v_entropy"] for kv in kv_cache_entropies]
            ),
        }

        print(f"\n{'KV Cache Statistics':^60}")
        print("─" * 60)
        print(
            f"Average KV Entropy:     {kv_analysis['average_kv_entropy']:8.6f} ± {kv_analysis['std_kv_entropy']:6.6f} bits"
        )
        print(
            f"Average K Entropy:      {kv_analysis['average_k_entropy']:8.6f} ± {kv_analysis['std_k_entropy']:6.6f} bits"
        )
        print(
            f"Average V Entropy:      {kv_analysis['average_v_entropy']:8.6f} ± {kv_analysis['std_v_entropy']:6.6f} bits"
        )

        # Save KV cache analysis
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        with open(os.path.join(output_dir, "kv_cache_analysis.json"), "w") as f:
            json.dump(convert_numpy(kv_analysis), f, indent=2)

        print(f"💾 KV Cache analysis saved: {output_dir}/kv_cache_analysis.json")

    # Save detailed parameter analysis
    parameter_analysis = {
        "model_size": model.get_num_params(),
        "parameter_entropies": entropies,
        "entropy_scaling": analyze_entropy_scaling(entropy_tracker),
    }

    with open(os.path.join(output_dir, "parameter_analysis.json"), "w") as f:
        json.dump(parameter_analysis, f, indent=2)

    print(f"💾 Parameter analysis saved: {output_dir}/parameter_analysis.json")

    return entropies


# ================================================================================================
# MAIN TRAINING FUNCTION (Similar to train.py main loop)
# ================================================================================================


def train_model(model_config, dataset_type, output_dir, max_iters=50000):
    """Train a model with entropy tracking"""

    print(f"\n{'🚀 TRAINING SESSION':=^80}")
    print(
        f"Model: {model_config} | Dataset: {dataset_type.upper()} | Output: {output_dir}"
    )
    print("=" * 80)

    # Setup device and precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    print(f"🖥️  Device: {device} | Precision: {dtype}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create model
    configs = create_model_configs()
    gptconf = configs[model_config]
    model = GPT(gptconf)
    model.to(device)

    print(f"🧠 Model created: {model.get_num_params():,} parameters")

    # Create data loader
    data_loader = DataLoader(
        dataset_type, vocab_size=gptconf.vocab_size, block_size=gptconf.block_size
    )

    # Setup optimizer (using standard GPT-2 configuration)
    learning_rate = 6e-4
    weight_decay = 1e-1
    beta1, beta2 = 0.9, 0.95
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device.split(":")[0]
    )

    # Setup entropy tracking
    entropy_tracker = EntropyTracker(num_bins=100)

    # Setup gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # Training parameters (similar to train.py defaults)
    batch_size = 12
    gradient_accumulation_steps = 5
    eval_interval = 1000
    log_interval = 100
    entropy_log_interval = 500
    warmup_iters = 2000
    lr_decay_iters = max_iters
    min_lr = 6e-5
    grad_clip = 1.0
    eval_iters = 100

    tokens_per_iter = gradient_accumulation_steps * batch_size * gptconf.block_size

    print("📋 Training Configuration:")
    print(f"   Max iterations:        {max_iters:,}")
    print(f"   Batch size:            {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Tokens per iteration:  {tokens_per_iter:,}")
    print(f"   Learning rate:         {learning_rate}")
    print(f"   Weight decay:          {weight_decay}")
    print(f"   Warmup iterations:     {warmup_iters:,}")

    # Training loop (similar structure to train.py)
    model.train()
    iter_num = 0
    best_val_loss = 1e9
    t0 = time.time()
    running_mfu = -1.0

    # Get first batch
    X, Y = data_loader.get_batch("train", batch_size)
    X, Y = X.to(device), Y.to(device)

    print("\n🎯 Starting training loop...")

    while iter_num < max_iters:
        # Determine learning rate for this iteration
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate and save checkpoint
        if iter_num % eval_interval == 0:
            try:
                losses = estimate_loss(model, data_loader, eval_iters, ctx, device)

                # Log entropy at evaluation intervals
                entropies = entropy_tracker.log_entropy(
                    model, iter_num, track_layers=False, track_components=True
                )

                print(
                    f"📊 Step {iter_num:6,}: train={losses['train']:.4f} | val={losses['val']:.4f} | "
                    f"entropy={entropies.get('entropy_total', 0):.4f} | lr={lr:.2e}"
                )

                # Save checkpoint if best validation loss
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": gptconf.__dict__,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "entropy_history": dict(entropy_tracker.entropy_history),
                        "entropy_iterations": entropy_tracker.iteration_history,
                    }
                    torch.save(checkpoint, os.path.join(output_dir, "best_model.pt"))
                    print(f"💾 New best model saved (val_loss: {best_val_loss:.4f})")
            except Exception as e:
                print(f"⚠️ Error during evaluation at iteration {iter_num}: {e}")
                # Continue training even if evaluation fails

        # Forward backward update with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps

            # Get next batch asynchronously
            X, Y = data_loader.get_batch("train", batch_size)
            X, Y = X.to(device), Y.to(device)

            # Backward pass
            scaler.scale(loss).backward()

        # Clip gradients
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Step optimizer
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % log_interval == 0:
            lossf = loss.item() * gradient_accumulation_steps
            if iter_num >= 5:
                mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"⚡ Iter {iter_num:6,}: loss={lossf:.4f} | time={dt*1000:5.1f}ms | mfu={running_mfu*100:4.1f}%"
            )

        # Log entropy at specified intervals
        if iter_num % entropy_log_interval == 0 and iter_num > 0:
            try:
                entropies = entropy_tracker.log_entropy(
                    model, iter_num, track_layers=False, track_components=False
                )
                print(
                    f"🔬 Iter {iter_num:6,}: entropy update - total: {entropies.get('entropy_total', 0):.4f} bits"
                )
            except Exception as e:
                print(f"⚠️ Error logging entropy at iteration {iter_num}: {e}")
                # Continue training even if entropy logging fails

        iter_num += 1

    print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")

    # Save all data files first (no plotting)
    print("\n💾 Saving all analysis data...")

    # Save raw entropy tracking data
    try:
        entropy_data = {
            "entropy_history": dict(entropy_tracker.entropy_history),
            "iteration_history": entropy_tracker.iteration_history,
            "num_bins": entropy_tracker.num_bins,
        }
        with open(os.path.join(output_dir, "entropy_tracking_data.json"), "w") as f:
            json.dump(entropy_data, f, indent=2)
        print(f"💾 Raw entropy data saved: {output_dir}/entropy_tracking_data.json")
    except Exception as e:
        print(f"❌ Error saving entropy tracking data: {e}")

    # Save entropy scaling analysis
    try:
        scaling_results = analyze_entropy_scaling(entropy_tracker)
        with open(os.path.join(output_dir, "entropy_scaling_analysis.json"), "w") as f:
            json.dump(scaling_results, f, indent=2)
        print(
            f"💾 Entropy scaling analysis saved: {output_dir}/entropy_scaling_analysis.json"
        )
    except Exception as e:
        print(f"❌ Error saving entropy scaling analysis: {e}")

    # Load best model for analysis (data only, no plots)
    try:
        checkpoint_path = os.path.join(output_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model"])

            # Perform detailed analysis of the trained model (data only)
            print("🔍 Performing model analysis (data only)...")
            analyze_trained_model_data_only(model, data_loader, device, ctx, output_dir)
        else:
            print(
                f"⚠️ No checkpoint found at {checkpoint_path}, skipping model analysis"
            )
    except Exception as e:
        print(f"❌ Error loading model or performing analysis: {e}")
        import traceback

        traceback.print_exc()

    print("✅ All data saved successfully! Plots will be generated separately.")
    return model, entropy_tracker


# ================================================================================================
# MAIN EXECUTION
# ================================================================================================


def print_experiment_header(
    experiment_num, total_experiments, model_config, dataset_type, experiment_name
):
    """Print a fancy experiment header"""
    print(f"\n{'':=^100}")
    print(
        f"{'🧪 EXPERIMENT ' + str(experiment_num) + '/' + str(total_experiments):^100}"
    )
    print(f"{'':=^100}")
    print(f"Name: {experiment_name}")
    print(f"Model: {model_config}")
    print(f"Dataset: {dataset_type}")
    print(f"{'':=^100}")


def generate_all_plots(base_output_dir, results):
    """Generate all plots from saved data after all experiments complete"""
    print(f"\n{'🎨 GENERATING ALL PLOTS FROM SAVED DATA':=^80}")

    for experiment_name, result in results.items():
        if result.get("status") == "failed":
            print(f"⚠️ Skipping plots for failed experiment: {experiment_name}")
            continue

        output_dir = result["output_dir"]
        print(f"\n🖼️ Generating plots for {experiment_name}...")

        # Generate entropy evolution plot
        try:
            entropy_data_path = os.path.join(output_dir, "entropy_tracking_data.json")
            if os.path.exists(entropy_data_path):
                # Load entropy data
                with open(entropy_data_path, "r") as f:
                    entropy_data = json.load(f)

                # Recreate entropy tracker from saved data
                entropy_tracker = EntropyTracker(
                    num_bins=entropy_data.get("num_bins", 100)
                )
                entropy_tracker.entropy_history = entropy_data["entropy_history"]
                entropy_tracker.iteration_history = entropy_data["iteration_history"]

                # Fix array length mismatch by ensuring both arrays have same length
                if "entropy_total" in entropy_tracker.entropy_history:
                    entropy_len = len(entropy_tracker.entropy_history["entropy_total"])
                    iter_len = len(entropy_tracker.iteration_history)
                    min_len = min(entropy_len, iter_len)

                    # Truncate both arrays to the same length
                    entropy_tracker.iteration_history = (
                        entropy_tracker.iteration_history[:min_len]
                    )
                    for key in entropy_tracker.entropy_history:
                        if isinstance(entropy_tracker.entropy_history[key], list):
                            entropy_tracker.entropy_history[key] = (
                                entropy_tracker.entropy_history[key][:min_len]
                            )

                    print(
                        f"   Fixed array lengths: entropy={entropy_len}, iter={iter_len}, using={min_len}"
                    )

                # Generate entropy evolution plot
                plot_path = os.path.join(output_dir, "entropy_evolution.png")
                entropy_tracker.plot_entropy_evolution(
                    save_path=plot_path, show_layers=False
                )

                # Also generate a detailed version with layers if layer data exists
                layer_keys = [
                    k
                    for k in entropy_tracker.entropy_history.keys()
                    if k.startswith("entropy_layer_")
                ]
                if layer_keys:
                    detailed_plot_path = os.path.join(
                        output_dir, "entropy_evolution_detailed.png"
                    )
                    entropy_tracker.plot_entropy_evolution(
                        save_path=detailed_plot_path, show_layers=True
                    )

            else:
                print(f"⚠️ No entropy tracking data found for {experiment_name}")

        except Exception as e:
            print(f"❌ Error generating entropy plots for {experiment_name}: {e}")

        # Generate comparative plots if we have parameter analysis data
        try:
            param_analysis_path = os.path.join(output_dir, "parameter_analysis.json")
            if os.path.exists(param_analysis_path):
                generate_parameter_analysis_plots(output_dir)
            else:
                print(f"⚠️ No parameter analysis data found for {experiment_name}")
        except Exception as e:
            print(
                f"❌ Error generating parameter analysis plots for {experiment_name}: {e}"
            )

    # Generate overall comparison plots
    try:
        generate_experiment_comparison_plots(base_output_dir, results)
    except Exception as e:
        print(f"❌ Error generating comparison plots: {e}")

    print(f"\n✅ All plots generated successfully!")


def generate_parameter_analysis_plots(output_dir):
    """Generate plots from parameter analysis data"""
    param_analysis_path = os.path.join(output_dir, "parameter_analysis.json")

    with open(param_analysis_path, "r") as f:
        param_data = json.load(f)

    entropies = param_data["parameter_entropies"]

    # Create parameter entropy bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot of different entropy components
    component_keys = [k for k in entropies.keys() if not k.startswith("entropy_layer_")]
    if component_keys:
        labels = [
            k.replace("entropy_", "").replace("_", " ").title() for k in component_keys
        ]
        values = [entropies[k] for k in component_keys]

        bars = ax1.bar(labels, values, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.set_title("Parameter Entropy by Component")
        ax1.set_ylabel("Entropy (bits)")
        ax1.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Layer entropy plot if available
    layer_keys = sorted(
        [k for k in entropies.keys() if k.startswith("entropy_layer_")],
        key=lambda x: int(x.replace("entropy_layer_", "")),
    )
    if layer_keys:
        layer_nums = [int(k.replace("entropy_layer_", "")) for k in layer_keys]
        layer_values = [entropies[k] for k in layer_keys]

        ax2.plot(
            layer_nums, layer_values, "o-", linewidth=2, markersize=6, color="orange"
        )
        ax2.set_title("Entropy by Layer")
        ax2.set_xlabel("Layer Number")
        ax2.set_ylabel("Entropy (bits)")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No layer data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Layer Entropy (No Data)")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "parameter_entropy_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Parameter analysis plot saved: {plot_path}")


def generate_experiment_comparison_plots(base_output_dir, results):
    """Generate comprehensive comparison plots across all experiments"""
    print("\n📈 Generating experiment comparison plots...")

    # Collect data from all successful experiments
    experiment_data = []
    entropy_evolution_data = {}  # Store entropy evolution for each experiment

    for exp_name, result in results.items():
        if result.get("status") == "failed":
            continue

        try:
            # Load entropy scaling data
            scaling_path = os.path.join(
                result["output_dir"], "entropy_scaling_analysis.json"
            )
            scaling_data = {}
            if os.path.exists(scaling_path):
                with open(scaling_path, "r") as f:
                    scaling_data = json.load(f)

            # Load entropy evolution data
            entropy_path = os.path.join(
                result["output_dir"], "entropy_tracking_data.json"
            )
            if os.path.exists(entropy_path):
                with open(entropy_path, "r") as f:
                    entropy_tracking = json.load(f)
                    entropy_evolution_data[exp_name] = entropy_tracking

            experiment_data.append(
                {
                    "name": exp_name,
                    "model_config": result["model_config"],
                    "dataset_type": result["dataset_type"],
                    "model_size": result["model_size"],
                    "final_entropy": result["final_entropy"],
                    "scaling_data": scaling_data,
                }
            )
        except Exception as e:
            print(f"⚠️ Could not load data for {exp_name}: {e}")

    if len(experiment_data) < 1:
        print("⚠️ No successful experiments to generate comparison plots")
        return

    # Generate multiple comprehensive comparison plot sets
    generate_basic_comparison_plots(base_output_dir, experiment_data)
    generate_entropy_evolution_comparison(
        base_output_dir, experiment_data, entropy_evolution_data
    )
    generate_performance_matrix_plot(base_output_dir, experiment_data)
    generate_detailed_analysis_plots(base_output_dir, experiment_data)

    # Generate comprehensive text reports
    generate_text_reports(base_output_dir, experiment_data, entropy_evolution_data)


def generate_basic_comparison_plots(base_output_dir, experiment_data):
    """Generate basic comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Basic Experiment Comparison Across Models and Datasets", fontsize=16)

    # Plot 1: Final entropy by model and dataset
    model_configs = list(set(exp["model_config"] for exp in experiment_data))
    dataset_types = list(set(exp["dataset_type"] for exp in experiment_data))

    for i, model_config in enumerate(model_configs):
        model_data = [
            exp for exp in experiment_data if exp["model_config"] == model_config
        ]
        datasets = [exp["dataset_type"] for exp in model_data]
        entropies = [exp["final_entropy"] for exp in model_data]

        x_pos = np.arange(len(datasets)) + i * 0.35
        axes[0, 0].bar(
            x_pos,
            entropies,
            0.35,
            label=model_config.replace("_", " ").title(),
            alpha=0.8,
        )

    axes[0, 0].set_title("Final Entropy by Model and Dataset")
    axes[0, 0].set_xlabel("Dataset Type")
    axes[0, 0].set_ylabel("Final Entropy (bits)")
    axes[0, 0].set_xticks(np.arange(len(dataset_types)) + 0.175)
    axes[0, 0].set_xticklabels([d.title() for d in dataset_types])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Model size vs final entropy
    model_sizes = [exp["model_size"] for exp in experiment_data]
    final_entropies = [exp["final_entropy"] for exp in experiment_data]
    colors = [
        "red"
        if "openwebtext" in exp["dataset_type"]
        else "blue"
        if "random" in exp["dataset_type"]
        else "green"
        for exp in experiment_data
    ]

    scatter = axes[0, 1].scatter(
        model_sizes, final_entropies, c=colors, alpha=0.7, s=100
    )
    axes[0, 1].set_title("Model Size vs Final Entropy")
    axes[0, 1].set_xlabel("Model Size (parameters)")
    axes[0, 1].set_ylabel("Final Entropy (bits)")
    axes[0, 1].grid(True, alpha=0.3)

    # Add legend for colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label="OpenWebText"),
        Patch(facecolor="blue", alpha=0.7, label="Random"),
        Patch(facecolor="green", alpha=0.7, label="Constant"),
    ]
    axes[0, 1].legend(handles=legend_elements)

    # Plot 3: Entropy change over training
    entropy_changes = [
        exp["scaling_data"].get("entropy_change", 0) for exp in experiment_data
    ]
    exp_names = [exp["name"].replace("_", "\n") for exp in experiment_data]

    bars = axes[1, 0].bar(
        range(len(exp_names)),
        entropy_changes,
        alpha=0.7,
        color=["red" if change > 0 else "blue" for change in entropy_changes],
    )
    axes[1, 0].set_title("Entropy Change During Training")
    axes[1, 0].set_xlabel("Experiments")
    axes[1, 0].set_ylabel("Entropy Change (bits)")
    axes[1, 0].set_xticks(range(len(exp_names)))
    axes[1, 0].set_xticklabels(exp_names, rotation=45, ha="right")
    axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Distribution of final entropies
    axes[1, 1].hist(
        final_entropies, bins=10, alpha=0.7, color="purple", edgecolor="black"
    )
    axes[1, 1].set_title("Distribution of Final Entropies")
    axes[1, 1].set_xlabel("Final Entropy (bits)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    # Add statistics
    mean_entropy = np.mean(final_entropies)
    std_entropy = np.std(final_entropies)
    axes[1, 1].axvline(
        mean_entropy, color="red", linestyle="--", label=f"Mean: {mean_entropy:.4f}"
    )
    axes[1, 1].legend()

    plt.tight_layout()
    comparison_plot_path = os.path.join(
        base_output_dir, "basic_experiment_comparison.png"
    )
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 Basic comparison plot saved: {comparison_plot_path}")


def generate_entropy_evolution_comparison(
    base_output_dir, experiment_data, entropy_evolution_data
):
    """Generate entropy evolution comparison across all experiments"""
    if not entropy_evolution_data:
        print("⚠️ No entropy evolution data available for comparison")
        return

    print("\n📈 Creating entropy evolution comparison plots...")

    # Create subplot for entropy evolution comparison
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Entropy Evolution Comparison Across All Experiments", fontsize=16)

    # Color schemes for different models and datasets
    model_colors = {"gpt2_half": "red", "small_full_vocab": "blue"}
    dataset_styles = {"openwebtext": "-", "random": "--", "constant": ":"}

    # Plot 1: All entropy evolutions together
    for exp_name, entropy_data in entropy_evolution_data.items():
        if "entropy_total" in entropy_data["entropy_history"]:
            try:
                iterations = entropy_data["iteration_history"]
                entropies = entropy_data["entropy_history"]["entropy_total"]

                # Find matching experiment data for styling
                exp_info = next(
                    (exp for exp in experiment_data if exp["name"] == exp_name), None
                )
                if exp_info:
                    model_config = exp_info["model_config"]
                    dataset_type = exp_info["dataset_type"]

                    color = model_colors.get(model_config, "gray")
                    style = dataset_styles.get(dataset_type, "-")

                    # Ensure arrays are same length
                    min_len = min(len(iterations), len(entropies))
                    if min_len > 0:
                        label = f"{model_config.replace('_', ' ').title()} - {dataset_type.title()}"
                        axes[0, 0].plot(
                            iterations[:min_len],
                            entropies[:min_len],
                            color=color,
                            linestyle=style,
                            label=label,
                            alpha=0.8,
                            linewidth=2,
                        )
            except Exception as e:
                print(f"⚠️ Error plotting entropy evolution for {exp_name}: {e}")

    axes[0, 0].set_title("All Entropy Evolution Curves")
    axes[0, 0].set_xlabel("Training Iteration")
    axes[0, 0].set_ylabel("Total Parameter Entropy (bits)")
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Final entropy heatmap
    model_configs = list(set(exp["model_config"] for exp in experiment_data))
    dataset_types = list(set(exp["dataset_type"] for exp in experiment_data))

    # Create matrix data
    entropy_matrix = np.full((len(model_configs), len(dataset_types)), np.nan)
    for i, model in enumerate(model_configs):
        for j, dataset in enumerate(dataset_types):
            matching_exp = next(
                (
                    exp
                    for exp in experiment_data
                    if exp["model_config"] == model and exp["dataset_type"] == dataset
                ),
                None,
            )
            if matching_exp:
                entropy_matrix[i, j] = matching_exp["final_entropy"]

    im = axes[0, 1].imshow(entropy_matrix, cmap="viridis", aspect="auto")
    axes[0, 1].set_title("Final Entropy Heatmap")
    axes[0, 1].set_xlabel("Dataset Type")
    axes[0, 1].set_ylabel("Model Configuration")
    axes[0, 1].set_xticks(range(len(dataset_types)))
    axes[0, 1].set_xticklabels([d.title() for d in dataset_types])
    axes[0, 1].set_yticks(range(len(model_configs)))
    axes[0, 1].set_yticklabels([m.replace("_", " ").title() for m in model_configs])

    # Add text annotations
    for i in range(len(model_configs)):
        for j in range(len(dataset_types)):
            if not np.isnan(entropy_matrix[i, j]):
                axes[0, 1].text(
                    j,
                    i,
                    f"{entropy_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

    plt.colorbar(im, ax=axes[0, 1], label="Final Entropy (bits)")

    # Plot 3: Entropy change during training
    exp_names = [exp["name"].replace("_", "\n") for exp in experiment_data]
    entropy_changes = []
    for exp in experiment_data:
        change = (
            exp["scaling_data"].get("entropy_change", 0) if exp["scaling_data"] else 0
        )
        entropy_changes.append(change)

    colors = ["red" if change < 0 else "green" for change in entropy_changes]
    bars = axes[1, 0].bar(
        range(len(exp_names)), entropy_changes, color=colors, alpha=0.7
    )
    axes[1, 0].set_title("Entropy Change During Training")
    axes[1, 0].set_xlabel("Experiments")
    axes[1, 0].set_ylabel("Entropy Change (bits)")
    axes[1, 0].set_xticks(range(len(exp_names)))
    axes[1, 0].set_xticklabels(exp_names, rotation=45, ha="right")
    axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, entropy_changes):
        if abs(value) > 0.001:  # Only show non-zero changes
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontweight="bold",
            )

    # Plot 4: Model size vs entropy relationship
    model_sizes = [exp["model_size"] for exp in experiment_data]
    final_entropies = [exp["final_entropy"] for exp in experiment_data]

    # Color by dataset type
    dataset_colors = {"openwebtext": "red", "random": "blue", "constant": "green"}
    colors = [
        dataset_colors.get(exp["dataset_type"], "gray") for exp in experiment_data
    ]

    scatter = axes[1, 1].scatter(
        model_sizes, final_entropies, c=colors, alpha=0.7, s=100
    )
    axes[1, 1].set_title("Model Size vs Final Entropy")
    axes[1, 1].set_xlabel("Model Size (parameters)")
    axes[1, 1].set_ylabel("Final Entropy (bits)")
    axes[1, 1].grid(True, alpha=0.3)

    # Add legend for datasets
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color, alpha=0.7, label=dataset.title())
        for dataset, color in dataset_colors.items()
        if dataset in dataset_types
    ]
    axes[1, 1].legend(handles=legend_elements)

    # Add trend line
    if len(model_sizes) > 1:
        z = np.polyfit(model_sizes, final_entropies, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(
            model_sizes,
            p(model_sizes),
            "r--",
            alpha=0.8,
            label=f"Trend: {z[0]:.2e}x + {z[1]:.3f}",
        )
        axes[1, 1].legend()

    plt.tight_layout()
    evolution_comparison_path = os.path.join(
        base_output_dir, "entropy_evolution_comparison.png"
    )
    plt.savefig(evolution_comparison_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 Entropy evolution comparison saved: {evolution_comparison_path}")


def generate_performance_matrix_plot(base_output_dir, experiment_data):
    """Generate performance matrix visualization"""
    print("\n📈 Creating performance matrix plot...")

    model_configs = list(set(exp["model_config"] for exp in experiment_data))
    dataset_types = list(set(exp["dataset_type"] for exp in experiment_data))

    if len(model_configs) < 1 or len(dataset_types) < 1:
        print("⚠️ Not enough variety in models/datasets for matrix plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model vs Dataset Performance Matrix", fontsize=16)

    # Matrix 1: Final Entropy
    entropy_matrix = np.full((len(model_configs), len(dataset_types)), np.nan)
    for i, model in enumerate(model_configs):
        for j, dataset in enumerate(dataset_types):
            matching_exp = next(
                (
                    exp
                    for exp in experiment_data
                    if exp["model_config"] == model and exp["dataset_type"] == dataset
                ),
                None,
            )
            if matching_exp:
                entropy_matrix[i, j] = matching_exp["final_entropy"]

    im1 = axes[0].imshow(entropy_matrix, cmap="viridis", aspect="auto")
    axes[0].set_title("Final Entropy (bits)")
    axes[0].set_xlabel("Dataset Type")
    axes[0].set_ylabel("Model Configuration")
    axes[0].set_xticks(range(len(dataset_types)))
    axes[0].set_xticklabels([d.title() for d in dataset_types])
    axes[0].set_yticks(range(len(model_configs)))
    axes[0].set_yticklabels([m.replace("_", " ").title() for m in model_configs])
    plt.colorbar(im1, ax=axes[0])

    # Add annotations
    for i in range(len(model_configs)):
        for j in range(len(dataset_types)):
            if not np.isnan(entropy_matrix[i, j]):
                axes[0].text(
                    j,
                    i,
                    f"{entropy_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

    # Matrix 2: Model Size
    size_matrix = np.full((len(model_configs), len(dataset_types)), np.nan)
    for i, model in enumerate(model_configs):
        for j, dataset in enumerate(dataset_types):
            matching_exp = next(
                (
                    exp
                    for exp in experiment_data
                    if exp["model_config"] == model and exp["dataset_type"] == dataset
                ),
                None,
            )
            if matching_exp:
                size_matrix[i, j] = matching_exp["model_size"] / 1e6  # In millions

    im2 = axes[1].imshow(size_matrix, cmap="plasma", aspect="auto")
    axes[1].set_title("Model Size (M parameters)")
    axes[1].set_xlabel("Dataset Type")
    axes[1].set_ylabel("Model Configuration")
    axes[1].set_xticks(range(len(dataset_types)))
    axes[1].set_xticklabels([d.title() for d in dataset_types])
    axes[1].set_yticks(range(len(model_configs)))
    axes[1].set_yticklabels([m.replace("_", " ").title() for m in model_configs])
    plt.colorbar(im2, ax=axes[1])

    # Add annotations
    for i in range(len(model_configs)):
        for j in range(len(dataset_types)):
            if not np.isnan(size_matrix[i, j]):
                axes[1].text(
                    j,
                    i,
                    f"{size_matrix[i, j]:.1f}M",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

    # Matrix 3: Entropy per Parameter (efficiency)
    efficiency_matrix = np.full((len(model_configs), len(dataset_types)), np.nan)
    for i, model in enumerate(model_configs):
        for j, dataset in enumerate(dataset_types):
            matching_exp = next(
                (
                    exp
                    for exp in experiment_data
                    if exp["model_config"] == model and exp["dataset_type"] == dataset
                ),
                None,
            )
            if matching_exp and matching_exp["model_size"] > 0:
                efficiency_matrix[i, j] = matching_exp["final_entropy"] / (
                    matching_exp["model_size"] / 1e6
                )

    im3 = axes[2].imshow(efficiency_matrix, cmap="coolwarm", aspect="auto")
    axes[2].set_title("Entropy Efficiency\n(entropy/M params)")
    axes[2].set_xlabel("Dataset Type")
    axes[2].set_ylabel("Model Configuration")
    axes[2].set_xticks(range(len(dataset_types)))
    axes[2].set_xticklabels([d.title() for d in dataset_types])
    axes[2].set_yticks(range(len(model_configs)))
    axes[2].set_yticklabels([m.replace("_", " ").title() for m in model_configs])
    plt.colorbar(im3, ax=axes[2])

    # Add annotations
    for i in range(len(model_configs)):
        for j in range(len(dataset_types)):
            if not np.isnan(efficiency_matrix[i, j]):
                axes[2].text(
                    j,
                    i,
                    f"{efficiency_matrix[i, j]:.4f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=8,
                )

    plt.tight_layout()
    matrix_path = os.path.join(base_output_dir, "performance_matrix.png")
    plt.savefig(matrix_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 Performance matrix saved: {matrix_path}")


def generate_detailed_analysis_plots(base_output_dir, experiment_data):
    """Generate detailed analysis plots"""
    print("\n📈 Creating detailed analysis plots...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Detailed Cross-Experiment Analysis", fontsize=16)

    # Plot 1: Final entropy distribution
    final_entropies = [exp["final_entropy"] for exp in experiment_data]
    axes[0, 0].hist(
        final_entropies,
        bins=min(10, len(set(final_entropies))),
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    axes[0, 0].set_title("Final Entropy Distribution")
    axes[0, 0].set_xlabel("Final Entropy (bits)")
    axes[0, 0].set_ylabel("Frequency")
    if len(final_entropies) > 0:
        axes[0, 0].axvline(
            np.mean(final_entropies),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(final_entropies):.4f}",
        )
        axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Model size distribution
    model_sizes = [exp["model_size"] / 1e6 for exp in experiment_data]  # In millions
    axes[0, 1].hist(
        model_sizes,
        bins=min(10, len(set(model_sizes))),
        alpha=0.7,
        color="lightgreen",
        edgecolor="black",
    )
    axes[0, 1].set_title("Model Size Distribution")
    axes[0, 1].set_xlabel("Model Size (M parameters)")
    axes[0, 1].set_ylabel("Frequency")
    if len(model_sizes) > 0:
        axes[0, 1].axvline(
            np.mean(model_sizes),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(model_sizes):.1f}M",
        )
        axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Dataset type comparison
    dataset_types = list(set(exp["dataset_type"] for exp in experiment_data))
    if len(dataset_types) > 1 and len(experiment_data) > 0:
        dataset_entropies = {}
        for dataset in dataset_types:
            dataset_entropies[dataset] = [
                exp["final_entropy"]
                for exp in experiment_data
                if exp["dataset_type"] == dataset
            ]

        box_data = [
            dataset_entropies[dataset]
            for dataset in dataset_types
            if dataset_entropies[dataset]
        ]
        if box_data:
            box_plot = axes[0, 2].boxplot(
                box_data,
                labels=[d.title() for d in dataset_types if dataset_entropies[d]],
            )
            axes[0, 2].set_title("Entropy by Dataset Type")
            axes[0, 2].set_xlabel("Dataset Type")
            axes[0, 2].set_ylabel("Final Entropy (bits)")
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(
                0.5,
                0.5,
                "No data\nfor comparison",
                ha="center",
                va="center",
                transform=axes[0, 2].transAxes,
            )
    else:
        axes[0, 2].text(
            0.5,
            0.5,
            "Insufficient\ndata variety",
            ha="center",
            va="center",
            transform=axes[0, 2].transAxes,
        )

    # Plot 4: Model config comparison
    model_configs = list(set(exp["model_config"] for exp in experiment_data))
    if len(model_configs) > 1 and len(experiment_data) > 0:
        model_entropies = {}
        for model in model_configs:
            model_entropies[model] = [
                exp["final_entropy"]
                for exp in experiment_data
                if exp["model_config"] == model
            ]

        box_data = [
            model_entropies[model] for model in model_configs if model_entropies[model]
        ]
        if box_data:
            axes[1, 0].boxplot(
                box_data,
                labels=[
                    m.replace("_", "\n") for m in model_configs if model_entropies[m]
                ],
            )
            axes[1, 0].set_title("Entropy by Model Configuration")
            axes[1, 0].set_xlabel("Model Configuration")
            axes[1, 0].set_ylabel("Final Entropy (bits)")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No data\nfor comparison",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Insufficient\nmodel variety",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )

    # Plot 5: Entropy vs Model Size correlation
    model_sizes_full = [exp["model_size"] for exp in experiment_data]
    final_entropies_full = [exp["final_entropy"] for exp in experiment_data]

    if len(model_sizes_full) > 1:
        # Calculate correlation
        correlation = np.corrcoef(model_sizes_full, final_entropies_full)[0, 1]

        axes[1, 1].scatter(model_sizes_full, final_entropies_full, alpha=0.7, s=100)
        axes[1, 1].set_title(f"Model Size vs Entropy\n(correlation: {correlation:.3f})")
        axes[1, 1].set_xlabel("Model Size (parameters)")
        axes[1, 1].set_ylabel("Final Entropy (bits)")
        axes[1, 1].grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(model_sizes_full, final_entropies_full, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(model_sizes_full, p(model_sizes_full), "r--", alpha=0.8)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Insufficient\ndata for\ncorrelation",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        correlation = 0.0

    # Plot 6: Summary statistics table
    axes[1, 2].axis("off")

    # Create summary statistics
    summary_stats = [
        ["Metric", "Value"],
        ["Total Experiments", str(len(experiment_data))],
        ["Model Configurations", str(len(model_configs))],
        ["Dataset Types", str(len(dataset_types))],
        ["", ""],
        ["Entropy Statistics", ""],
    ]

    if final_entropies:
        summary_stats.extend(
            [
                ["Mean Final Entropy", f"{np.mean(final_entropies):.6f} bits"],
                ["Std Final Entropy", f"{np.std(final_entropies):.6f} bits"],
                ["Min Final Entropy", f"{np.min(final_entropies):.6f} bits"],
                ["Max Final Entropy", f"{np.max(final_entropies):.6f} bits"],
            ]
        )

    summary_stats.extend(
        [
            ["", ""],
            ["Model Size Statistics", ""],
        ]
    )

    if model_sizes:
        summary_stats.extend(
            [
                ["Mean Model Size", f"{np.mean(model_sizes):.1f}M params"],
                ["Std Model Size", f"{np.std(model_sizes):.1f}M params"],
                ["Size-Entropy Correlation", f"{correlation:.3f}"],
            ]
        )

    # Create table
    table = axes[1, 2].table(
        cellText=summary_stats,
        cellColours=[["lightgray", "lightgray"]]
        + [["white", "white"]] * (len(summary_stats) - 1),
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axes[1, 2].set_title("Summary Statistics")

    plt.tight_layout()
    detailed_path = os.path.join(base_output_dir, "detailed_analysis.png")
    plt.savefig(detailed_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 Detailed analysis plot saved: {detailed_path}")


def generate_text_reports(base_output_dir, experiment_data, entropy_evolution_data):
    """Generate comprehensive text reports for all analyses"""
    print("\n📝 Generating comprehensive text reports...")

    # Generate main analysis report
    generate_main_analysis_report(
        base_output_dir, experiment_data, entropy_evolution_data
    )

    # Generate individual experiment summaries
    generate_individual_experiment_reports(
        base_output_dir, experiment_data, entropy_evolution_data
    )

    # Generate comparative analysis report
    generate_comparative_analysis_report(base_output_dir, experiment_data)

    # Generate entropy evolution analysis
    generate_entropy_evolution_report(
        base_output_dir, experiment_data, entropy_evolution_data
    )

    print("✅ All text reports generated successfully!")


def generate_main_analysis_report(
    base_output_dir, experiment_data, entropy_evolution_data
):
    """Generate the main comprehensive analysis report"""
    report_path = os.path.join(base_output_dir, "COMPREHENSIVE_ANALYSIS_REPORT.md")

    with open(report_path, "w") as f:
        f.write("# Comprehensive nanoGPT Entropy Analysis Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(
            f"This report analyzes entropy behavior across {len(experiment_data)} experiments "
        )
        f.write(
            f"involving {len(set(exp['model_config'] for exp in experiment_data))} model configurations "
        )
        f.write(
            f"and {len(set(exp['dataset_type'] for exp in experiment_data))} dataset types.\n\n"
        )

        if experiment_data:
            final_entropies = [exp["final_entropy"] for exp in experiment_data]
            f.write(f"**Key Findings:**\n")
            f.write(
                f"- Final entropy range: {min(final_entropies):.6f} to {max(final_entropies):.6f} bits\n"
            )
            f.write(
                f"- Mean final entropy: {np.mean(final_entropies):.6f} ± {np.std(final_entropies):.6f} bits\n"
            )

            # Find best and worst performers
            best_exp = min(experiment_data, key=lambda x: x["final_entropy"])
            worst_exp = max(experiment_data, key=lambda x: x["final_entropy"])
            f.write(
                f"- Best performing: {best_exp['name']} ({best_exp['final_entropy']:.6f} bits)\n"
            )
            f.write(
                f"- Worst performing: {worst_exp['name']} ({worst_exp['final_entropy']:.6f} bits)\n\n"
            )

        # Detailed Results Table
        f.write("## Detailed Results\n\n")
        f.write(
            "| Experiment | Model Config | Dataset | Model Size (M) | Final Entropy (bits) | Entropy Change |\n"
        )
        f.write(
            "|-----------|-------------|---------|---------------|---------------------|---------------|\n"
        )

        for exp in sorted(experiment_data, key=lambda x: x["final_entropy"]):
            model_size_m = exp["model_size"] / 1e6
            entropy_change = (
                exp["scaling_data"].get("entropy_change", 0)
                if exp["scaling_data"]
                else 0
            )
            f.write(
                f"| {exp['name']} | {exp['model_config']} | {exp['dataset_type']} | "
            )
            f.write(
                f"{model_size_m:.1f} | {exp['final_entropy']:.6f} | {entropy_change:+.6f} |\n"
            )

        # Model Configuration Analysis
        f.write("\n## Model Configuration Analysis\n\n")
        model_configs = list(set(exp["model_config"] for exp in experiment_data))

        for model_config in model_configs:
            model_exps = [
                exp for exp in experiment_data if exp["model_config"] == model_config
            ]
            if model_exps:
                model_entropies = [exp["final_entropy"] for exp in model_exps]
                f.write(f"### {model_config.replace('_', ' ').title()}\n\n")
                f.write(f"- Experiments: {len(model_exps)}\n")
                f.write(
                    f"- Model size: {model_exps[0]['model_size'] / 1e6:.1f}M parameters\n"
                )
                f.write(
                    f"- Entropy range: {min(model_entropies):.6f} to {max(model_entropies):.6f} bits\n"
                )
                f.write(
                    f"- Mean entropy: {np.mean(model_entropies):.6f} ± {np.std(model_entropies):.6f} bits\n"
                )

                # Best dataset for this model
                best_dataset_exp = min(model_exps, key=lambda x: x["final_entropy"])
                f.write(
                    f"- Best dataset: {best_dataset_exp['dataset_type']} ({best_dataset_exp['final_entropy']:.6f} bits)\n\n"
                )

        # Dataset Analysis
        f.write("## Dataset Analysis\n\n")
        dataset_types = list(set(exp["dataset_type"] for exp in experiment_data))

        for dataset_type in dataset_types:
            dataset_exps = [
                exp for exp in experiment_data if exp["dataset_type"] == dataset_type
            ]
            if dataset_exps:
                dataset_entropies = [exp["final_entropy"] for exp in dataset_exps]
                f.write(f"### {dataset_type.title()} Dataset\n\n")
                f.write(f"- Experiments: {len(dataset_exps)}\n")
                f.write(
                    f"- Entropy range: {min(dataset_entropies):.6f} to {max(dataset_entropies):.6f} bits\n"
                )
                f.write(
                    f"- Mean entropy: {np.mean(dataset_entropies):.6f} ± {np.std(dataset_entropies):.6f} bits\n"
                )

                # Best model for this dataset
                best_model_exp = min(dataset_exps, key=lambda x: x["final_entropy"])
                f.write(
                    f"- Best model: {best_model_exp['model_config']} ({best_model_exp['final_entropy']:.6f} bits)\n\n"
                )

        # Statistical Analysis
        if len(experiment_data) > 1:
            f.write("## Statistical Analysis\n\n")

            # Correlation between model size and entropy
            model_sizes = [exp["model_size"] for exp in experiment_data]
            final_entropies = [exp["final_entropy"] for exp in experiment_data]

            if len(set(model_sizes)) > 1:  # Only if we have different model sizes
                correlation = np.corrcoef(model_sizes, final_entropies)[0, 1]
                f.write(f"### Model Size vs Entropy\n\n")
                f.write(f"- Correlation coefficient: {correlation:.4f}\n")

                if abs(correlation) > 0.7:
                    strength = "strong"
                elif abs(correlation) > 0.3:
                    strength = "moderate"
                else:
                    strength = "weak"

                direction = "positive" if correlation > 0 else "negative"
                f.write(f"- Relationship: {strength} {direction} correlation\n")

                if correlation > 0:
                    f.write(
                        "- Interpretation: Larger models tend to have higher entropy\n\n"
                    )
                else:
                    f.write(
                        "- Interpretation: Larger models tend to have lower entropy\n\n"
                    )

            # ANOVA-style analysis if we have multiple groups
            if len(model_configs) > 1:
                f.write(f"### Model Configuration Comparison\n\n")

                # Calculate between-group and within-group variance
                overall_mean = np.mean(final_entropies)

                between_group_variance = 0
                within_group_variance = 0
                total_n = 0

                for model_config in model_configs:
                    model_entropies = [
                        exp["final_entropy"]
                        for exp in experiment_data
                        if exp["model_config"] == model_config
                    ]
                    if len(model_entropies) > 0:
                        group_mean = np.mean(model_entropies)
                        n = len(model_entropies)
                        total_n += n

                        between_group_variance += n * (group_mean - overall_mean) ** 2
                        within_group_variance += sum(
                            (x - group_mean) ** 2 for x in model_entropies
                        )

                if within_group_variance > 0:
                    f_ratio = (between_group_variance / (len(model_configs) - 1)) / (
                        within_group_variance / (total_n - len(model_configs))
                    )
                    f.write(f"- F-ratio: {f_ratio:.4f}\n")

                    if f_ratio > 4.0:  # Rough threshold for significance
                        f.write(
                            "- Model configurations show significantly different entropy patterns\n\n"
                        )
                    else:
                        f.write(
                            "- Model configurations show similar entropy patterns\n\n"
                        )

        # Key Insights
        f.write("## Key Insights\n\n")

        if experiment_data:
            # Find patterns
            insights = []

            # Dataset performance patterns
            if len(dataset_types) > 1:
                dataset_means = {}
                for dataset in dataset_types:
                    dataset_entropies = [
                        exp["final_entropy"]
                        for exp in experiment_data
                        if exp["dataset_type"] == dataset
                    ]
                    if dataset_entropies:
                        dataset_means[dataset] = np.mean(dataset_entropies)

                if dataset_means:
                    best_dataset = min(dataset_means.items(), key=lambda x: x[1])
                    worst_dataset = max(dataset_means.items(), key=lambda x: x[1])

                    insights.append(
                        f"**Dataset Performance**: {best_dataset[0].title()} dataset consistently produces "
                    )
                    insights.append(
                        f"lower entropy ({best_dataset[1]:.6f} bits avg) compared to {worst_dataset[0].title()} "
                    )
                    insights.append(f"dataset ({worst_dataset[1]:.6f} bits avg).")

            # Model size efficiency
            if len(model_configs) > 1:
                efficiency_scores = {}
                for model_config in model_configs:
                    model_exps = [
                        exp
                        for exp in experiment_data
                        if exp["model_config"] == model_config
                    ]
                    if model_exps:
                        avg_entropy = np.mean(
                            [exp["final_entropy"] for exp in model_exps]
                        )
                        model_size = model_exps[0]["model_size"] / 1e6
                        efficiency_scores[model_config] = avg_entropy / model_size

                if efficiency_scores:
                    most_efficient = min(efficiency_scores.items(), key=lambda x: x[1])
                    insights.append(
                        f"\n\n**Model Efficiency**: {most_efficient[0].replace('_', ' ').title()} "
                    )
                    insights.append(f"shows the best entropy-per-parameter efficiency ")
                    insights.append(f"({most_efficient[1]:.6f} bits/M params).")

            # Entropy change patterns
            entropy_changes = [
                exp["scaling_data"].get("entropy_change", 0)
                for exp in experiment_data
                if exp["scaling_data"]
            ]
            if entropy_changes:
                decreasing = sum(1 for change in entropy_changes if change < -0.001)
                increasing = sum(1 for change in entropy_changes if change > 0.001)
                stable = len(entropy_changes) - decreasing - increasing

                insights.append(
                    f"\n\n**Training Dynamics**: {decreasing} experiments showed entropy decrease, "
                )
                insights.append(
                    f"{increasing} showed increase, {stable} remained stable during training."
                )

            for insight in insights:
                f.write(insight)

        f.write("\n\n## Files Generated\n\n")
        f.write("This analysis generated the following visualization files:\n\n")
        f.write("- `basic_experiment_comparison.png` - Overview comparison charts\n")
        f.write(
            "- `entropy_evolution_comparison.png` - Training dynamics across experiments\n"
        )
        f.write(
            "- `performance_matrix.png` - Heatmap matrices of performance metrics\n"
        )
        f.write("- `detailed_analysis.png` - Statistical analysis and distributions\n")
        f.write("- Individual experiment plots in each experiment directory\n\n")

        f.write(
            "For detailed methodology and implementation details, see the source code and individual experiment logs.\n"
        )

    print(f"📝 Main analysis report saved: {report_path}")


def print_final_summary(results, base_output_dir, start_time):
    """Print comprehensive final summary"""
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{'':=^100}")
    print(f"{'🎉 EXPERIMENT COMPLETE':^100}")
    print(f"{'':=^100}")

    print("📊 SUMMARY STATISTICS")
    print(f"{'─' * 50}")
    print(f"Total experiments:     {len(results)}")
    print(
        f"Total runtime:         {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)"
    )
    print(f"Average per experiment: {total_time/len(results)/60:.1f} minutes")
    print(f"Output directory:      {base_output_dir}")

    # Group results by model configuration
    print("\n📈 RESULTS BY MODEL CONFIGURATION")
    print(f"{'─' * 80}")

    model_groups = {}
    for exp_name, result in results.items():
        model_config = result["model_config"]
        if model_config not in model_groups:
            model_groups[model_config] = []
        model_groups[model_config].append(result)

    for model_config, experiments in model_groups.items():
        print(
            f"\n🧠 {model_config.upper().replace('_', ' ')} ({experiments[0]['model_size']:,} parameters)"
        )
        for exp in experiments:
            dataset = exp["dataset_type"].upper()
            entropy = exp["final_entropy"]
            print(f"   {dataset:12s}: entropy = {entropy:8.6f} bits")

    # Group results by dataset type
    print("\n📊 RESULTS BY DATASET TYPE")
    print(f"{'─' * 80}")

    dataset_groups = {}
    for exp_name, result in results.items():
        dataset_type = result["dataset_type"]
        if dataset_type not in dataset_groups:
            dataset_groups[dataset_type] = []
        dataset_groups[dataset_type].append(result)

    for dataset_type, experiments in dataset_groups.items():
        print(f"\n📁 {dataset_type.upper()} DATASET")
        for exp in experiments:
            model = exp["model_config"].replace("_", " ").title()
            entropy = exp["final_entropy"]
            size = exp["model_size"]
            print(f"   {model:20s}: {size:8,} params, entropy = {entropy:8.6f} bits")

    # Entropy analysis
    print("\n🔬 ENTROPY ANALYSIS")
    print(f"{'─' * 80}")

    all_entropies = [result["final_entropy"] for result in results.values()]
    if all_entropies:
        print(f"Highest entropy: {max(all_entropies):8.6f} bits")
        print(f"Lowest entropy:  {min(all_entropies):8.6f} bits")
        print(f"Average entropy: {np.mean(all_entropies):8.6f} bits")
        print(f"Std deviation:   {np.std(all_entropies):8.6f} bits")

    # Files generated
    print("\n📁 FILES GENERATED")
    print(f"{'─' * 50}")
    print("📋 experiment_summary.json - Overall results")
    for exp_name in results.keys():
        print(f"📂 {exp_name}/")
        print("   ├── 🤖 best_model.pt - Trained model checkpoint")
        print("   ├── 📊 entropy_evolution.png - Training entropy plot")
        print("   ├── 📈 entropy_scaling_analysis.json - Scaling metrics")
        print("   ├── 🔍 parameter_analysis.json - Component analysis")
        print("   └── 💾 kv_cache_analysis.json - KV cache entropy")

    print(f"\n{'':=^100}")
    print(f"✨ All analyses complete! Check {base_output_dir} for detailed results.")
    print(f"{'':=^100}")


def main():
    """Main execution function"""
    start_time = time.time()

    print("🚀 COMPREHENSIVE NANOGPT ENTROPY ANALYSIS")
    print("=" * 100)
    print("Training multiple model configurations on different datasets")
    print("with comprehensive entropy tracking and analysis.")
    print("=" * 100)

    # Create base output directory
    base_output_dir = "entropy_experiments"
    os.makedirs(base_output_dir, exist_ok=True)

    # Define experiments
    experiments = [
        # GPT-2 Half size on different datasets
        ("gpt2_half", "openwebtext", "gpt2_half_openwebtext"),
        ("gpt2_half", "random", "gpt2_half_random"),
        ("gpt2_half", "constant", "gpt2_half_constant"),
        # Small model with full vocab on different datasets
        ("small_full_vocab", "openwebtext", "small_full_vocab_openwebtext"),
        ("small_full_vocab", "random", "small_full_vocab_random"),
        ("small_full_vocab", "constant", "small_full_vocab_constant"),
    ]

    results = {}

    for i, (model_config, dataset_type, experiment_name) in enumerate(experiments, 1):
        print_experiment_header(
            i, len(experiments), model_config, dataset_type, experiment_name
        )

        output_dir = os.path.join(base_output_dir, experiment_name)

        # Check if experiment already completed (for resuming)
        checkpoint_path = os.path.join(output_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            print(
                f"🔄 Found existing checkpoint for {experiment_name}, loading results..."
            )
            try:
                # Load existing results
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )

                # Create a minimal entropy tracker from saved data
                entropy_tracker = EntropyTracker(num_bins=100)
                if "entropy_history" in checkpoint:
                    entropy_tracker.entropy_history = checkpoint["entropy_history"]
                if "entropy_iterations" in checkpoint:
                    entropy_tracker.iteration_history = checkpoint["entropy_iterations"]

                # Store results from existing checkpoint
                results[experiment_name] = {
                    "model_config": model_config,
                    "dataset_type": dataset_type,
                    "output_dir": output_dir,
                    "final_entropy": entropy_tracker.get_latest_entropy().get(
                        "entropy_total", 0.0
                    ),
                    "model_size": checkpoint.get("model_args", {}).get("n_embd", 0)
                    * checkpoint.get("model_args", {}).get("n_layer", 0),
                    "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
                }

                print(f"✅ Experiment {i}/{len(experiments)} loaded from checkpoint!")
                continue

            except Exception as e:
                print(f"⚠️ Warning: Could not load checkpoint {checkpoint_path}: {e}")
                print("Starting fresh training...")

        try:
            # Train model
            model, entropy_tracker = train_model(
                model_config=model_config,
                dataset_type=dataset_type,
                output_dir=output_dir,
                max_iters=20000,  # Reduced for faster experimentation, increase for full training
            )

            # Store results
            results[experiment_name] = {
                "model_config": model_config,
                "dataset_type": dataset_type,
                "output_dir": output_dir,
                "final_entropy": entropy_tracker.get_latest_entropy().get(
                    "entropy_total", 0.0
                ),
                "model_size": model.get_num_params(),
                "best_val_loss": float("inf"),  # Would be populated from checkpoint
            }

            print(f"✅ Experiment {i}/{len(experiments)} completed successfully!")

        except Exception as e:
            print(f"❌ Error in experiment {experiment_name}: {e}")
            import traceback

            traceback.print_exc()

            # Save partial results with error info
            results[experiment_name] = {
                "model_config": model_config,
                "dataset_type": dataset_type,
                "output_dir": output_dir,
                "final_entropy": 0.0,
                "model_size": 0,
                "best_val_loss": float("inf"),
                "error": str(e),
                "status": "failed",
            }

            print(
                f"⚠️ Experiment {i}/{len(experiments)} failed, continuing with next experiment..."
            )
            continue

    # Save overall results summary
    summary_data = {
        "experiments": results,
        "timestamp": time.time(),
        "total_experiments": len(experiments),
        "successful_experiments": len(results),
        "total_runtime_hours": (time.time() - start_time) / 3600,
        "configurations": {
            "max_iters_per_experiment": 20000,
            "entropy_bins": 100,
            "model_configs": ["gpt2_half", "small_full_vocab"],
            "dataset_types": ["openwebtext", "random", "constant"],
        },
    }

    with open(os.path.join(base_output_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    print("✅ All experiment data saved successfully!")

    # Generate all plots from saved data
    print("\n" + "=" * 80)
    print("Now generating all plots from the saved data...")
    print("=" * 80)

    generate_all_plots(base_output_dir, results)

    # Print comprehensive final summary
    print_final_summary(results, base_output_dir, start_time)


def generate_individual_experiment_reports(
    base_output_dir, experiment_data, entropy_evolution_data
):
    """Generate individual text reports for each experiment"""
    reports_dir = os.path.join(base_output_dir, "text_reports")
    os.makedirs(reports_dir, exist_ok=True)

    for exp in experiment_data:
        exp_name = exp["name"]
        report_path = os.path.join(reports_dir, f"{exp_name}_report.txt")

        with open(report_path, "w") as f:
            f.write(f"EXPERIMENT REPORT: {exp_name.upper()}\n")
            f.write("=" * 60 + "\n\n")

            # Basic Information
            f.write("CONFIGURATION:\n")
            f.write(f"  Model Configuration: {exp['model_config']}\n")
            f.write(f"  Dataset Type: {exp['dataset_type']}\n")
            f.write(
                f"  Model Size: {exp['model_size']:,} parameters ({exp['model_size']/1e6:.1f}M)\n\n"
            )

            # Results
            f.write("RESULTS:\n")
            f.write(f"  Final Entropy: {exp['final_entropy']:.6f} bits\n")

            if exp["scaling_data"]:
                scaling = exp["scaling_data"]
                f.write(
                    f"  Initial Entropy: {scaling.get('initial_entropy', 'N/A'):.6f} bits\n"
                )
                f.write(
                    f"  Entropy Change: {scaling.get('entropy_change', 0):+.6f} bits\n"
                )
                f.write(
                    f"  Mean Change Rate: {scaling.get('mean_change_rate', 0):.8f} bits/iter\n"
                )
                f.write(
                    f"  Max Entropy: {scaling.get('max_entropy', 'N/A'):.6f} bits\n"
                )
                f.write(
                    f"  Min Entropy: {scaling.get('min_entropy', 'N/A'):.6f} bits\n"
                )
                f.write(
                    f"  Entropy Stability (CV): {scaling.get('entropy_coefficient_variation', 'N/A'):.6f}\n\n"
                )

            # Entropy Evolution Analysis
            if exp_name in entropy_evolution_data:
                entropy_data = entropy_evolution_data[exp_name]
                if "entropy_total" in entropy_data["entropy_history"]:
                    entropy_series = entropy_data["entropy_history"]["entropy_total"]
                    iterations = entropy_data["iteration_history"]

                    min_len = min(len(entropy_series), len(iterations))
                    if min_len > 0:
                        entropy_series = entropy_series[:min_len]
                        iterations = iterations[:min_len]

                        f.write("ENTROPY EVOLUTION ANALYSIS:\n")
                        f.write(f"  Data Points: {len(entropy_series)}\n")
                        f.write(
                            f"  Training Iterations: {iterations[0]} to {iterations[-1]}\n"
                        )
                        f.write(
                            f"  Entropy Range: {min(entropy_series):.6f} to {max(entropy_series):.6f} bits\n"
                        )

                        # Trend analysis
                        if len(entropy_series) > 1:
                            # Simple linear trend
                            x = np.arange(len(entropy_series))
                            slope, intercept = np.polyfit(x, entropy_series, 1)

                            f.write(
                                f"  Linear Trend Slope: {slope:.8f} bits/measurement\n"
                            )

                            if abs(slope) < 1e-6:
                                trend = "stable"
                            elif slope > 0:
                                trend = "increasing"
                            else:
                                trend = "decreasing"

                            f.write(f"  Overall Trend: {trend}\n")

                            # Volatility (standard deviation)
                            volatility = np.std(entropy_series)
                            f.write(f"  Entropy Volatility: {volatility:.6f} bits\n\n")

                        # Key timepoints
                        f.write("KEY TIMEPOINTS:\n")
                        quarter_points = [
                            0,
                            len(entropy_series) // 4,
                            len(entropy_series) // 2,
                            3 * len(entropy_series) // 4,
                            len(entropy_series) - 1,
                        ]
                        labels = ["Start", "25%", "50%", "75%", "End"]

                        for i, (point, label) in enumerate(zip(quarter_points, labels)):
                            if point < len(entropy_series):
                                f.write(
                                    f"  {label:5s} (iter {iterations[point]:6,}): {entropy_series[point]:.6f} bits\n"
                                )

                        f.write("\n")

            # Performance Ranking
            all_entropies = [e["final_entropy"] for e in experiment_data]
            rank = sorted(all_entropies).index(exp["final_entropy"]) + 1
            f.write("PERFORMANCE RANKING:\n")
            f.write(f"  Rank: {rank} out of {len(experiment_data)} experiments\n")
            f.write(
                f"  Percentile: {((len(experiment_data) - rank) / len(experiment_data) * 100):.1f}th percentile\n\n"
            )

            # Comparison with same model/dataset
            same_model = [
                e for e in experiment_data if e["model_config"] == exp["model_config"]
            ]
            same_dataset = [
                e for e in experiment_data if e["dataset_type"] == exp["dataset_type"]
            ]

            if len(same_model) > 1:
                model_entropies = [e["final_entropy"] for e in same_model]
                model_rank = sorted(model_entropies).index(exp["final_entropy"]) + 1
                f.write(f"WITHIN MODEL CONFIG ({exp['model_config']}):\n")
                f.write(f"  Rank: {model_rank} out of {len(same_model)}\n")
                f.write(
                    f"  Model Config Average: {np.mean(model_entropies):.6f} bits\n\n"
                )

            if len(same_dataset) > 1:
                dataset_entropies = [e["final_entropy"] for e in same_dataset]
                dataset_rank = sorted(dataset_entropies).index(exp["final_entropy"]) + 1
                f.write(f"WITHIN DATASET ({exp['dataset_type']}):\n")
                f.write(f"  Rank: {dataset_rank} out of {len(same_dataset)}\n")
                f.write(f"  Dataset Average: {np.mean(dataset_entropies):.6f} bits\n\n")

    print(f"📝 Individual experiment reports saved in: {reports_dir}")


def generate_comparative_analysis_report(base_output_dir, experiment_data):
    """Generate comparative analysis text report"""
    report_path = os.path.join(base_output_dir, "comparative_analysis.txt")

    with open(report_path, "w") as f:
        f.write("COMPARATIVE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        # Model Configuration Comparison
        f.write("MODEL CONFIGURATION COMPARISON:\n")
        f.write("-" * 35 + "\n")

        model_configs = list(set(exp["model_config"] for exp in experiment_data))
        for model_config in sorted(model_configs):
            model_exps = [
                exp for exp in experiment_data if exp["model_config"] == model_config
            ]
            if model_exps:
                entropies = [exp["final_entropy"] for exp in model_exps]
                f.write(f"\n{model_config.upper().replace('_', ' ')}:\n")
                f.write(f"  Experiments: {len(model_exps)}\n")
                f.write(
                    f"  Model Size: {model_exps[0]['model_size']/1e6:.1f}M parameters\n"
                )
                f.write(
                    f"  Mean Entropy: {np.mean(entropies):.6f} ± {np.std(entropies):.6f} bits\n"
                )
                f.write(f"  Range: {min(entropies):.6f} to {max(entropies):.6f} bits\n")

                # Best and worst datasets for this model
                best_exp = min(model_exps, key=lambda x: x["final_entropy"])
                worst_exp = max(model_exps, key=lambda x: x["final_entropy"])
                f.write(
                    f"  Best Dataset: {best_exp['dataset_type']} ({best_exp['final_entropy']:.6f} bits)\n"
                )
                f.write(
                    f"  Worst Dataset: {worst_exp['dataset_type']} ({worst_exp['final_entropy']:.6f} bits)\n"
                )

        # Dataset Type Comparison
        f.write("\n\nDATASET TYPE COMPARISON:\n")
        f.write("-" * 25 + "\n")

        dataset_types = list(set(exp["dataset_type"] for exp in experiment_data))
        for dataset_type in sorted(dataset_types):
            dataset_exps = [
                exp for exp in experiment_data if exp["dataset_type"] == dataset_type
            ]
            if dataset_exps:
                entropies = [exp["final_entropy"] for exp in dataset_exps]
                f.write(f"\n{dataset_type.upper()}:\n")
                f.write(f"  Experiments: {len(dataset_exps)}\n")
                f.write(
                    f"  Mean Entropy: {np.mean(entropies):.6f} ± {np.std(entropies):.6f} bits\n"
                )
                f.write(f"  Range: {min(entropies):.6f} to {max(entropies):.6f} bits\n")

                # Best and worst models for this dataset
                best_exp = min(dataset_exps, key=lambda x: x["final_entropy"])
                worst_exp = max(dataset_exps, key=lambda x: x["final_entropy"])
                f.write(
                    f"  Best Model: {best_exp['model_config']} ({best_exp['final_entropy']:.6f} bits)\n"
                )
                f.write(
                    f"  Worst Model: {worst_exp['model_config']} ({worst_exp['final_entropy']:.6f} bits)\n"
                )

        # Performance Matrix
        f.write("\n\nPERFORMANCE MATRIX (Final Entropy in bits):\n")
        f.write("-" * 45 + "\n\n")

        # Create a table
        f.write(f"{'Model Config':<20} ")
        for dataset in sorted(dataset_types):
            f.write(f"{dataset:<12} ")
        f.write("\n" + "-" * (20 + 12 * len(dataset_types)) + "\n")

        for model in sorted(model_configs):
            f.write(f"{model:<20} ")
            for dataset in sorted(dataset_types):
                matching_exp = next(
                    (
                        exp
                        for exp in experiment_data
                        if exp["model_config"] == model
                        and exp["dataset_type"] == dataset
                    ),
                    None,
                )
                if matching_exp:
                    f.write(f"{matching_exp['final_entropy']:<12.6f} ")
                else:
                    f.write(f"{'N/A':<12} ")
            f.write("\n")

        # Statistical Summary
        f.write("\n\nSTATISTICAL SUMMARY:\n")
        f.write("-" * 20 + "\n")

        all_entropies = [exp["final_entropy"] for exp in experiment_data]
        f.write(f"Total Experiments: {len(experiment_data)}\n")
        f.write(f"Overall Mean: {np.mean(all_entropies):.6f} bits\n")
        f.write(f"Overall Std Dev: {np.std(all_entropies):.6f} bits\n")
        f.write(
            f"Overall Range: {min(all_entropies):.6f} to {max(all_entropies):.6f} bits\n"
        )
        f.write(
            f"Coefficient of Variation: {np.std(all_entropies)/np.mean(all_entropies):.4f}\n"
        )

        # Best and worst overall
        best_overall = min(experiment_data, key=lambda x: x["final_entropy"])
        worst_overall = max(experiment_data, key=lambda x: x["final_entropy"])
        f.write(
            f"\nBest Overall: {best_overall['name']} ({best_overall['final_entropy']:.6f} bits)\n"
        )
        f.write(
            f"Worst Overall: {worst_overall['name']} ({worst_overall['final_entropy']:.6f} bits)\n"
        )
        f.write(
            f"Performance Spread: {worst_overall['final_entropy'] - best_overall['final_entropy']:.6f} bits\n"
        )

    print(f"📝 Comparative analysis report saved: {report_path}")


def generate_entropy_evolution_report(
    base_output_dir, experiment_data, entropy_evolution_data
):
    """Generate entropy evolution analysis report"""
    if not entropy_evolution_data:
        return

    report_path = os.path.join(base_output_dir, "entropy_evolution_analysis.txt")

    with open(report_path, "w") as f:
        f.write("ENTROPY EVOLUTION ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")

        f.write(
            "This report analyzes how entropy changes during training across all experiments.\n\n"
        )

        # Overall evolution patterns
        f.write("OVERALL EVOLUTION PATTERNS:\n")
        f.write("-" * 30 + "\n")

        evolution_stats = []

        for exp_name, entropy_data in entropy_evolution_data.items():
            if "entropy_total" in entropy_data["entropy_history"]:
                entropy_series = entropy_data["entropy_history"]["entropy_total"]
                iterations = entropy_data["iteration_history"]

                min_len = min(len(entropy_series), len(iterations))
                if min_len > 1:
                    entropy_series = entropy_series[:min_len]

                    # Calculate trend
                    x = np.arange(len(entropy_series))
                    slope, _ = np.polyfit(x, entropy_series, 1)

                    # Calculate volatility
                    volatility = np.std(entropy_series)

                    # Get experiment info
                    exp_info = next(
                        (exp for exp in experiment_data if exp["name"] == exp_name),
                        None,
                    )

                    evolution_stats.append(
                        {
                            "name": exp_name,
                            "model_config": exp_info["model_config"]
                            if exp_info
                            else "unknown",
                            "dataset_type": exp_info["dataset_type"]
                            if exp_info
                            else "unknown",
                            "slope": slope,
                            "volatility": volatility,
                            "initial_entropy": entropy_series[0],
                            "final_entropy": entropy_series[-1],
                            "data_points": len(entropy_series),
                        }
                    )

        if evolution_stats:
            # Sort by slope to see trends
            evolution_stats.sort(key=lambda x: x["slope"])

            f.write(
                f"Total experiments with evolution data: {len(evolution_stats)}\n\n"
            )

            # Trend categories
            decreasing = [stat for stat in evolution_stats if stat["slope"] < -1e-6]
            stable = [stat for stat in evolution_stats if abs(stat["slope"]) <= 1e-6]
            increasing = [stat for stat in evolution_stats if stat["slope"] > 1e-6]

            f.write(f"Trend Distribution:\n")
            f.write(
                f"  Decreasing entropy: {len(decreasing)} experiments ({len(decreasing)/len(evolution_stats)*100:.1f}%)\n"
            )
            f.write(
                f"  Stable entropy: {len(stable)} experiments ({len(stable)/len(evolution_stats)*100:.1f}%)\n"
            )
            f.write(
                f"  Increasing entropy: {len(increasing)} experiments ({len(increasing)/len(evolution_stats)*100:.1f}%)\n\n"
            )

            # Detailed breakdown
            if decreasing:
                f.write("DECREASING ENTROPY EXPERIMENTS:\n")
                for stat in decreasing:
                    f.write(f"  {stat['name']}: {stat['slope']:.8f} bits/step ")
                    f.write(
                        f"({stat['initial_entropy']:.6f} → {stat['final_entropy']:.6f} bits)\n"
                    )
                f.write("\n")

            if increasing:
                f.write("INCREASING ENTROPY EXPERIMENTS:\n")
                for stat in increasing:
                    f.write(f"  {stat['name']}: {stat['slope']:.8f} bits/step ")
                    f.write(
                        f"({stat['initial_entropy']:.6f} → {stat['final_entropy']:.6f} bits)\n"
                    )
                f.write("\n")

            if stable:
                f.write("STABLE ENTROPY EXPERIMENTS:\n")
                for stat in stable:
                    f.write(f"  {stat['name']}: {stat['slope']:.8f} bits/step ")
                    f.write(f"(volatility: {stat['volatility']:.6f} bits)\n")
                f.write("\n")

            # Volatility analysis
            f.write("VOLATILITY ANALYSIS:\n")
            f.write("-" * 20 + "\n")

            volatilities = [stat["volatility"] for stat in evolution_stats]
            f.write(f"Mean volatility: {np.mean(volatilities):.6f} bits\n")
            f.write(
                f"Volatility range: {min(volatilities):.6f} to {max(volatilities):.6f} bits\n\n"
            )

            # Most/least volatile
            most_volatile = max(evolution_stats, key=lambda x: x["volatility"])
            least_volatile = min(evolution_stats, key=lambda x: x["volatility"])

            f.write(
                f"Most volatile: {most_volatile['name']} ({most_volatile['volatility']:.6f} bits)\n"
            )
            f.write(
                f"Least volatile: {least_volatile['name']} ({least_volatile['volatility']:.6f} bits)\n\n"
            )

            # Model/Dataset pattern analysis
            f.write("PATTERN ANALYSIS BY MODEL CONFIGURATION:\n")
            f.write("-" * 45 + "\n")

            model_configs = list(set(stat["model_config"] for stat in evolution_stats))
            for model_config in model_configs:
                model_stats = [
                    stat
                    for stat in evolution_stats
                    if stat["model_config"] == model_config
                ]
                if model_stats:
                    model_slopes = [stat["slope"] for stat in model_stats]
                    model_volatilities = [stat["volatility"] for stat in model_stats]

                    f.write(f"\n{model_config.upper().replace('_', ' ')}:\n")
                    f.write(f"  Experiments: {len(model_stats)}\n")
                    f.write(f"  Mean slope: {np.mean(model_slopes):.8f} bits/step\n")
                    f.write(
                        f"  Mean volatility: {np.mean(model_volatilities):.6f} bits\n"
                    )

                    dec_count = sum(1 for slope in model_slopes if slope < -1e-6)
                    inc_count = sum(1 for slope in model_slopes if slope > 1e-6)
                    sta_count = len(model_slopes) - dec_count - inc_count

                    f.write(
                        f"  Trends: {dec_count} decreasing, {sta_count} stable, {inc_count} increasing\n"
                    )

            f.write("\n\nPATTERN ANALYSIS BY DATASET TYPE:\n")
            f.write("-" * 35 + "\n")

            dataset_types = list(set(stat["dataset_type"] for stat in evolution_stats))
            for dataset_type in dataset_types:
                dataset_stats = [
                    stat
                    for stat in evolution_stats
                    if stat["dataset_type"] == dataset_type
                ]
                if dataset_stats:
                    dataset_slopes = [stat["slope"] for stat in dataset_stats]
                    dataset_volatilities = [
                        stat["volatility"] for stat in dataset_stats
                    ]

                    f.write(f"\n{dataset_type.upper()}:\n")
                    f.write(f"  Experiments: {len(dataset_stats)}\n")
                    f.write(f"  Mean slope: {np.mean(dataset_slopes):.8f} bits/step\n")
                    f.write(
                        f"  Mean volatility: {np.mean(dataset_volatilities):.6f} bits\n"
                    )

                    dec_count = sum(1 for slope in dataset_slopes if slope < -1e-6)
                    inc_count = sum(1 for slope in dataset_slopes if slope > 1e-6)
                    sta_count = len(dataset_slopes) - dec_count - inc_count

                    f.write(
                        f"  Trends: {dec_count} decreasing, {sta_count} stable, {inc_count} increasing\n"
                    )

    print(f"📝 Entropy evolution report saved: {report_path}")


def generate_plots_only(base_output_dir="entropy_experiments"):
    """Generate plots from existing data without running experiments"""
    if not os.path.exists(base_output_dir):
        print(f"❌ Error: Output directory {base_output_dir} does not exist")
        return

    # Load experiment summary
    summary_path = os.path.join(base_output_dir, "experiment_summary.json")
    if not os.path.exists(summary_path):
        print(f"❌ Error: No experiment summary found at {summary_path}")
        return

    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    results = summary_data.get("experiments", {})

    if not results:
        print("❌ Error: No experiment results found in summary")
        return

    print(f"🎨 Generating plots for {len(results)} experiments...")
    generate_all_plots(base_output_dir, results)
    print("✅ Plot generation complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--plots-only":
        # Generate plots from existing data
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "entropy_experiments"
        generate_plots_only(output_dir)
    else:
        # Run full experiments
        main()
