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

    def plot_entropy_evolution(self, save_path=None, show_layers=False):
        """Plot entropy evolution over training"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Parameter Entropy Evolution During Training")

        # Plot total entropy
        if "entropy_total" in self.entropy_history:
            axes[0, 0].plot(
                self.iteration_history,
                self.entropy_history["entropy_total"],
                "b-",
                linewidth=2,
            )
            axes[0, 0].set_title("Total Parameter Entropy")
            axes[0, 0].set_xlabel("Iteration")
            axes[0, 0].set_ylabel("Entropy (bits)")
            axes[0, 0].grid(True)

        # Plot component entropies if available
        component_keys = [
            k
            for k in self.entropy_history.keys()
            if k.startswith("entropy_")
            and k != "entropy_total"
            and not k.startswith("entropy_layer_")
        ]
        if component_keys:
            for key in component_keys:
                label = key.replace("entropy_", "").replace("_", " ").title()
                axes[0, 1].plot(
                    self.iteration_history,
                    self.entropy_history[key],
                    label=label,
                    linewidth=2,
                )
            axes[0, 1].set_title("Component Entropy")
            axes[0, 1].set_xlabel("Iteration")
            axes[0, 1].set_ylabel("Entropy (bits)")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # Plot layer entropies if available
        layer_keys = [
            k for k in self.entropy_history.keys() if k.startswith("entropy_layer_")
        ]
        if layer_keys and show_layers:
            for key in layer_keys:
                layer_num = key.replace("entropy_layer_", "")
                axes[1, 0].plot(
                    self.iteration_history,
                    self.entropy_history[key],
                    label=f"Layer {layer_num}",
                    linewidth=1.5,
                )
            axes[1, 0].set_title("Layer-wise Entropy")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Entropy (bits)")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Plot entropy distribution
        if "entropy_total" in self.entropy_history:
            axes[1, 1].hist(
                self.entropy_history["entropy_total"],
                bins=20,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            axes[1, 1].set_title("Total Entropy Distribution")
            axes[1, 1].set_xlabel("Entropy (bits)")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Entropy evolution plot saved: {save_path}")
        else:
            plt.show()

        plt.close()


def analyze_entropy_scaling(entropy_tracker):
    """Analyze entropy scaling properties"""
    if (
        "entropy_total" not in entropy_tracker.entropy_history
        or len(entropy_tracker.entropy_history["entropy_total"]) < 10
    ):
        return {}

    entropies = np.array(entropy_tracker.entropy_history["entropy_total"])
    iterations = np.array(entropy_tracker.iteration_history)

    # Calculate entropy change rate
    if len(entropies) > 1:
        entropy_change_rate = np.diff(entropies) / np.diff(iterations)
        mean_change_rate = np.mean(entropy_change_rate)
        std_change_rate = np.std(entropy_change_rate)
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
            entropies = entropy_tracker.log_entropy(
                model, iter_num, track_layers=False, track_components=False
            )
            print(
                f"🔬 Iter {iter_num:6,}: entropy update - total: {entropies.get('entropy_total', 0):.4f} bits"
            )

        iter_num += 1

    print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")

    # Save entropy scaling analysis first (data)
    try:
        scaling_results = analyze_entropy_scaling(entropy_tracker)
        with open(os.path.join(output_dir, "entropy_scaling_analysis.json"), "w") as f:
            json.dump(scaling_results, f, indent=2)
        print(f"💾 Entropy scaling analysis saved: {output_dir}/entropy_scaling_analysis.json")
    except Exception as e:
        print(f"❌ Error saving entropy scaling analysis: {e}")

    # Load best model for analysis
    try:
        checkpoint = torch.load(
            os.path.join(output_dir, "best_model.pt"), map_location=device
        )
        model.load_state_dict(checkpoint["model"])
        
        # Perform detailed analysis of the trained model
        analyze_trained_model(model, data_loader, device, ctx, output_dir)
    except Exception as e:
        print(f"❌ Error loading model or performing analysis: {e}")

    # Save plots at the very end after all data is written
    try:
        plot_path = os.path.join(output_dir, "entropy_evolution.png")
        entropy_tracker.plot_entropy_evolution(save_path=plot_path, show_layers=False)
    except Exception as e:
        print(f"❌ Error creating final entropy plot: {e}")

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

    # Print comprehensive final summary
    print_final_summary(results, base_output_dir, start_time)


if __name__ == "__main__":
    main()
