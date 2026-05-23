#!/usr/bin/env python3
"""
train_hrm_text_pi.py — Train an HRM-Text prompt injection detector
on the Bordair multimodal dataset with ~128k context support.

Architecture follows HRM-Text (sapientinc/HRM-Text, arXiv:2506.21734):
  - ScaledEmbeddingInit: byte-level embedding with lecun scaling
  - H module: high-level transformer stack (recurrent)
  - L module: low-level transformer stack (recurrent)
  - Recurrent loop: H_cycles × (L_cycles × L → H) cascade
  - Classification head on final z_H (last-token pooling)
  - RoPE with optional NTK-aware scaling (default 8k, configurable up to 128k)
  - Backprop warmup: 2→5 recurrent steps over first 20% of training

Data: Bordair/bordair-multimodal (503K samples, balanced 1:1)
Target: ~36M parameters
"""
import os
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import glob
import time
from collections import Counter

import datasets as hf_datasets
import evaluate
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from datasets import Dataset, concatenate_datasets
from huggingface_hub import snapshot_download, HfApi
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Embedding (NTK-aware scaling for 128k)
# ═══════════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """RoPE with optional NTK-aware scaling for long context extension.

    Standard RoPE: theta=10000.0, max_seq_len=4096.
    NTK scaling: scale_factor = target_len / original_max_len.
    Extends context by redistributing frequencies.
    """

    def __init__(self, dim, max_seq_len=4096, base=10000.0, scaling_factor=32.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        if scaling_factor > 1.0:
            # NTK-aware scaling: adjust base instead of interpolating positions
            ntk_base = base * scaling_factor ** (dim / (dim - 2))
        else:
            ntk_base = base

        inv_freq = 1.0 / (ntk_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids):
        cos = self.cos_cached[position_ids]  # [B, L, dim]
        sin = self.sin_cached[position_ids]
        return cos, sin


def apply_rotary_pos_emb(x, cos, sin):
    """Apply RoPE to tensor x [B, L, H, HD] using precomputed cos/sin."""
    half = x.shape[-1] // 2
    x_rot = x[..., :half].to(cos.dtype)
    x_pass = x[..., half:].to(cos.dtype)
    cos = cos[..., :half].unsqueeze(-2)
    sin = sin[..., :half].unsqueeze(-2)
    x_rot_out = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([x_rot_out, x_pass], dim=-1)


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Initialization helpers
# ═══════════════════════════════════════════════════════════════════════════════

def trunc_normal_init_(tensor, std=1.0):
    """Truncated normal approximation via 3-sigma clamping."""
    with torch.no_grad():
        return tensor.normal_().fmod_(3.0).mul_(1.014762601732121 * std)


class LinearInit(nn.Linear):
    """Linear layer with lecun-normal-like init."""

    def __init__(self, in_features, out_features, bias=True, init_std=None):
        super().__init__(in_features, out_features, bias=bias)
        if init_std is None:
            init_std = 1.0 / math.sqrt(in_features)
        trunc_normal_init_(self.weight, std=init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


# ═══════════════════════════════════════════════════════════════════════════════
# Scaled Embedding
# ═══════════════════════════════════════════════════════════════════════════════

class ScaledEmbeddingInit(nn.Embedding):
    """Embedding with lecun-normal scaling (HRM-Text §2.1)."""

    def __init__(self, vocab_size, d_model, padding_idx=0, init_std=None):
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)
        if init_std is None:
            init_std = 1.0 / math.sqrt(d_model)
        trunc_normal_init_(self.weight, std=init_std)
        with torch.no_grad():
            if padding_idx is not None:
                self.weight[padding_idx].zero_()
        self.scale = 1.0 / init_std if init_std > 0 else 1.0

    def forward(self, input_ids):
        return super().forward(input_ids) * self.scale


# ═══════════════════════════════════════════════════════════════════════════════
# Gated Attention (HRM-Text sigmoid-gated MHA)
# ═══════════════════════════════════════════════════════════════════════════════

class GatedAttention(nn.Module):
    """Sigmoid-gated multi-head attention with RoPE.

    Single projection: gate + q + k + v → split → gate(sigmoid) × attn → o_proj
    """

    def __init__(self, hidden_size, num_heads, head_dim, init_std):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Single projection for gate (G), query (Q), key (K), value (V)
        # G: num_heads, Q: num_heads, K: num_heads, V: num_heads
        n_gqkv = num_heads * 4  # G+Q+K+V
        self.gqkv_proj = LinearInit(
            hidden_size, head_dim * n_gqkv,
            bias=False, init_std=init_std,
        )
        self.o_proj = LinearInit(
            head_dim * num_heads, hidden_size,
            bias=False, init_std=init_std,
        )

    def forward(self, hidden_states, cos, sin):
        B, L, D = hidden_states.shape

        gqkv = self.gqkv_proj(hidden_states)
        gqkv = gqkv.view(B, L, self.num_heads * 4, self.head_dim)

        gate, query, key, value = gqkv.split(self.num_heads, dim=2)
        # gate: [B, L, num_heads, head_dim]
        # query, key, value: [B, L, num_heads, head_dim]

        # RoPE on Q and K
        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)

        # Transpose to [B, num_heads, L, head_dim] for SDPA
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # SDPA with causal masking (flash attention backend, O(L) memory)
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            is_causal=True,
        )

        # Gate: sigmoid(gate) × attn_output (elementwise)
        gate = gate.transpose(1, 2)  # [B, num_heads, L, head_dim]
        attn_output = torch.sigmoid(gate) * attn_output

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(attn_output)


# ═══════════════════════════════════════════════════════════════════════════════
# SwiGLU FFN
# ═══════════════════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: gate_up_proj → SiLU(gate)*up → down_proj.

    gate_up_proj maps hidden_size → intermediate_size * 2 (gate + up).
    """

    def __init__(self, hidden_size, intermediate_size, init_std):
        super().__init__()
        self.gate_up_proj = LinearInit(
            hidden_size, intermediate_size * 2,
            bias=False, init_std=init_std,
        )
        self.down_proj = LinearInit(
            intermediate_size, hidden_size,
            bias=False, init_std=init_std,
        )

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# ═══════════════════════════════════════════════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Attn → Residual → SwiGLU → Residual."""

    def __init__(self, hidden_size, num_heads, head_dim, intermediate_size, init_std):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = GatedAttention(hidden_size, num_heads, head_dim, init_std)
        self.ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ffn = SwiGLU(hidden_size, intermediate_size, init_std)

    def forward(self, x, cos, sin):
        # Pre-norm attention (causal via SDPA)
        x = x + self.attn(self.attn_norm(x), cos, sin)
        # Pre-norm FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Recurrent Module (H or L)
# ═══════════════════════════════════════════════════════════════════════════════

class RecurrentModule(nn.Module):
    """A stack of TransformerBlocks used as one recurrent module (H or L).

    In HRM-Text, each module is a full transformer stack. The module receives
    its own hidden state + the other module's hidden state (via additive fusion).
    """

    def __init__(self, layers, hidden_size, num_heads, head_dim, intermediate_size, init_std, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, head_dim, intermediate_size, init_std)
            for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, z_self, z_other, cos, sin):
        """Forward through all blocks with additive cross-module fusion.

        Args:
            z_self: This module's hidden state [B, L, D]
            z_other: Other module's hidden state [B, L, D]
        """
        x = z_self + z_other  # Additive fusion
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch_checkpoint(block, x, cos, sin, use_reentrant=True)
            else:
                x = block(x, cos, sin)
        x = self.final_norm(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# HRM-Text Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class HrmTextClassifier(nn.Module):
    """
    HRM-Text adapted for binary classification (prompt injection detection).

    Architecture:
      f_emb: ScaledEmbeddingInit (byte-level, vocab=256)
      L_module: RecurrentModule (low-level transformer stack)
      H_module: RecurrentModule (high-level transformer stack)
      f_cls: Classification head (LayerNorm → Linear)

    Recurrent loop (H_cycles × L_cycles):
      z_L = L_module(z_L, z_H, cos, sin)
      z_H = H_module(z_H, z_L, cos, sin)

    Final: logits = f_cls(z_H[:, -1, :])  # last-token pooling for classification

    Backprop warmup: gradient-track only the last bp_steps recurrent steps.
    """

    def __init__(
        self,
        vocab_size=256,
        hidden_size=768,
        num_heads=12,
        head_dim=64,
        n_layers_H=3,
        n_layers_L=3,
        intermediate_size=2048,
        H_cycles=2,
        L_cycles=3,
        max_seq_len=4096,
        rope_base=10000.0,
        rope_scaling_factor=32.0,
        num_classes=2,
        bp_min_steps=2,
        bp_max_steps=5,
        bp_warmup_ratio=0.2,
        use_gradient_checkpointing=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.bp_min_steps = bp_min_steps
        self.bp_max_steps = bp_max_steps
        self.bp_warmup_ratio = bp_warmup_ratio
        self.total_steps = H_cycles * L_cycles  # total recurrent steps

        init_std = 1.0 / math.sqrt(hidden_size)  # lecun-normal std

        # Token embedding (byte-level)
        self.embed = ScaledEmbeddingInit(
            vocab_size, hidden_size, padding_idx=0,
            init_std=init_std,
        )

        # z_L initial state (learned buffer)
        self.zL_init = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Rotary embeddings (NTK-scaled for 128k)
        self.rotary = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
            scaling_factor=rope_scaling_factor,
        )

        # Recurrent modules
        self.L_module = RecurrentModule(
            layers=n_layers_L,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            init_std=init_std,
            use_checkpoint=use_gradient_checkpointing,
        )
        self.H_module = RecurrentModule(
            layers=n_layers_H,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            init_std=init_std,
            use_checkpoint=use_gradient_checkpointing,
        )

        # Classification head (on last token of z_H)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        # zL_init small
        nn.init.zeros_(self.zL_init)
        # Classifier init
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                trunc_normal_init_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_bp_steps(self, training_step_ratio=1.0):
        """Compute number of backprop steps (warmup from bp_min to bp_max)."""
        if training_step_ratio >= 1.0:
            return min(self.bp_max_steps, self.total_steps)
        warmup_progress = min(1.0, training_step_ratio / self.bp_warmup_ratio)
        bp = self.bp_min_steps + warmup_progress * (self.bp_max_steps - self.bp_min_steps)
        return min(int(bp), self.total_steps)

    def forward(self, input_ids, attention_mask=None, labels=None, training_step_ratio=None):
        """
        Args:
            input_ids: [B, L] byte token IDs
            attention_mask: [B, L] 1=valid, 0=padding
            labels: [B] binary labels (0=safe, 1=injection)
            training_step_ratio: float in [0, 1] for BP warmup scheduling
        Returns:
            dict with logits and optional loss
        """
        B, L = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        # Position IDs
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        # Apply attention mask to position IDs (positions after padding clamped)
        position_ids = position_ids * attention_mask

        # ── Embedding ──
        z_H = self.embed(input_ids)  # [B, L, D]
        z_L = self.zL_init.expand(B, L, -1)  # [B, L, D]

        # ── RoPE ──
        cos, sin = self.rotary(position_ids)  # [B, L, head_dim]

        # ── Attention: use is_causal=True with flash attention ──
        # At 128k context, explicit attention masks are prohibitively large
        # (17B elements = 34GB). Causal flash attention uses O(L) memory.
        # Padding tokens after the sequence naturally don't affect the
        # last valid token's representation under causal masking.

        # ── BP warmup ──
        bp_steps = self._get_bp_steps(training_step_ratio or 1.0)
        H_bp = min(self.H_cycles, max(1, bp_steps - 1))
        L_bp = max(0, bp_steps - H_bp)
        # Map: last L_bp L-steps across all cycles, last H_bp H-steps
        # Each H-cycle has L_cycles L-steps inside it
        total_L_steps = self.H_cycles * self.L_cycles
        total_H_steps = self.H_cycles

        # ── Recurrent loop ──
        # BP warmup: block gradient flow through early steps via .detach()
        # instead of torch.set_grad_enabled (incompatible with torch.compile)
        step_idx = 0
        for i in range(self.H_cycles):
            for k in range(self.L_cycles):
                grad_enabled = (step_idx >= total_L_steps - L_bp)
                z_L = self.L_module(
                    z_L if grad_enabled else z_L.detach(),
                    z_H if grad_enabled else z_H.detach(),
                    cos, sin,
                )
                step_idx += 1

            H_grad_enabled = (i >= self.H_cycles - H_bp)
            z_H = self.H_module(
                z_H if H_grad_enabled else z_H.detach(),
                z_L if H_grad_enabled else z_L.detach(),
                cos, sin,
            )

        # ── Classification: pool from the last valid token of each sequence ──
        # Use last-token pooling: grab the final non-padding token's representation
        # Find last valid position
        seq_lengths = attention_mask.sum(dim=1).long()  # [B]
        last_token_indices = (seq_lengths - 1).clamp(min=0)  # [B]
        batch_indices = torch.arange(B, device=device)
        pooled = z_H[batch_indices, last_token_indices, :]  # [B, D]

        logits = self.classifier(pooled)  # [B, num_classes]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels)

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def inference(self, input_ids, attention_mask=None):
        """Inference-only forward (all steps with gradients disabled)."""
        self.eval()
        B, L = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        position_ids = position_ids * attention_mask

        z_H = self.embed(input_ids)
        z_L = self.zL_init.expand(B, L, -1)
        cos, sin = self.rotary(position_ids)

        for _ in range(self.H_cycles):
            for _ in range(self.L_cycles):
                z_L = self.L_module(z_L, z_H, cos, sin)
            z_H = self.H_module(z_H, z_L, cos, sin)

        seq_lengths = attention_mask.sum(dim=1).long()
        last_token_indices = (seq_lengths - 1).clamp(min=0)
        pooled = z_H[torch.arange(B, device=device), last_token_indices, :]
        logits = self.classifier(pooled)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# Data pipeline — Bordair multimodal loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_bordair_multimodal(cache_dir=None, max_samples=None):
    """Load the full Bordair multimodal dataset from HF Hub.

    The dataset is stored as raw JSON arrays (not HF Dataset format).
    We snapshot_download the repo and read all JSON files manually.

    Returns:
        Dataset with columns: "text" (concatenated modalities), "label" (0/1)
    """
    print("📦 Downloading Bordair/bordair-multimodal from HF Hub...")
    path = snapshot_download(
        repo_id="Bordair/bordair-multimodal",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    print(f"   Downloaded to: {path}")

    all_samples = []

    # Pattern: collect all JSON files, skip summary/pool metadata
    dir_patterns = [
        "benign/*.json",
        "payloads/*/*.json",
        "payloads_v5/*.json",
        "payloads_v5_external/*/*.json",
    ]

    for pattern in dir_patterns:
        files = sorted(glob.glob(os.path.join(path, pattern)))
        for f in files:
            fname = os.path.basename(f)
            if fname in ("summary.json", "_pool.json", "summary_old.json"):
                continue

            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"   ⚠️  Skipping {f}: {e}")
                continue

            if not isinstance(data, list):
                continue

            for item in data:
                if not isinstance(item, dict):
                    continue
                all_samples.append(item)

        print(f"   {pattern}: {len(all_samples)} cumulative")

    print(f"\n✅ Total raw samples loaded: {len(all_samples)}")

    # Build unified dataset
    # Concatenate all text fields: text + image_content + document_content + audio_content
    rows = []
    labels = []
    skipped = 0
    for item in all_samples:
        # Get expected_detection (the boolean label)
        label_val = item.get("expected_detection")
        if label_val is None:
            skipped += 1
            continue

        text_parts = []
        if item.get("text"):
            text_parts.append(item["text"])
        if item.get("image_content"):
            text_parts.append(item["image_content"])
        if item.get("document_content"):
            text_parts.append(item["document_content"])
        if item.get("audio_content"):
            text_parts.append(item["audio_content"])

        combined = "\n".join(text_parts)

        # Skip empty texts
        if not combined.strip():
            skipped += 1
            continue

        rows.append(combined)
        labels.append(1 if label_val else 0)

    if skipped:
        print(f"   Skipped {skipped} samples (missing label or empty text)")

    # Convert to HF Dataset
    ds = Dataset.from_dict({"text": rows, "label": labels})
    print(f"✅ Dataset: {len(ds)} samples ({sum(labels)} injection, {len(labels) - sum(labels)} safe)")
    return ds


def normalize_label(ex, label_col):
    """Convert label to int64 0/1 (for compatibility with other datasets)."""
    val = ex[label_col]
    if isinstance(val, str):
        return {label_col: 1 if val.lower() in ("malicious", "injection", "yes", "1") else 0}
    return {label_col: int(val)}


# ═══════════════════════════════════════════════════════════════════════════════
# Byte-level tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

class ByteTokenizer:
    """Byte-level tokenizer: encodes strings as byte IDs [0-255].

    Supports variable-length sequences — padded in collation, not here.
    """

    def __init__(self, max_length=131072):
        self.max_length = max_length
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<pad>"
        self.vocab_size = 256

    def __call__(self, text, truncation=True, max_length=None):
        max_len = max_length or self.max_length
        if isinstance(text, str):
            byte_ids = list(text.encode("utf-8", errors="replace"))
        else:
            byte_ids = []
        if truncation:
            byte_ids = byte_ids[:max_len]
        return byte_ids

    def encode_batch(self, texts, max_length=None):
        """Encode a batch of texts into variable-length byte ID lists."""
        max_len = max_length or self.max_length
        result = []
        for text in texts:
            if isinstance(text, str):
                byte_ids = list(text.encode("utf-8", errors="replace"))
            else:
                byte_ids = []
            if max_len:
                byte_ids = byte_ids[:max_len]
            result.append(byte_ids)
        return result

    def __len__(self):
        return self.vocab_size


def collate_hrm_text(batch, max_length=131072):
    """Collation for HRM-Text: variable-length byte sequences with padding.

    Returns dict with input_ids, attention_mask, labels.
    Sequences are padded to the max length in the batch (not to max_length).
    """
    texts = [ex["text"] for ex in batch]
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)

    # Encode to byte IDs
    all_ids = []
    for t in texts:
        ids = list(t.encode("utf-8", errors="replace")[:max_length])
        all_ids.append(ids)

    max_len_in_batch = max(len(ids) for ids in all_ids) if all_ids else 0
    # Clamp to prevent excessive padding
    max_len_in_batch = min(max_len_in_batch, max_length)

    all_ids_padded = []
    attention_masks = []
    for ids in all_ids:
        length = min(len(ids), max_length)
        ids = ids[:length]
        padded = ids + [0] * (max_len_in_batch - length)
        mask = [1] * length + [0] * (max_len_in_batch - length)
        all_ids_padded.append(padded)
        attention_masks.append(mask)

    return {
        "input_ids": torch.tensor(all_ids_padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": labels,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class HrmTextTrainer(Trainer):
    """Trainer subclass that handles HRM-Text's custom forward signature
    and BP warmup scheduling."""

    def __init__(self, *args, total_training_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_training_steps = total_training_steps
        self._current_step = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # Compute training step ratio for BP warmup
        if self.total_training_steps and self.total_training_steps > 0:
            step_ratio = min(1.0, self._current_step / self.total_training_steps)
        else:
            step_ratio = 1.0

        outputs = model(**inputs, labels=labels, training_step_ratio=step_ratio)
        loss = outputs["loss"]
        self._current_step += 1
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        labels = inputs.pop("labels") if "labels" in inputs else None
        with torch.no_grad():
            logits = model.inference(**inputs)
        if prediction_loss_only:
            return (None, None, labels)
        return (None, logits, labels)


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter counting
# ═══════════════════════════════════════════════════════════════════════════════

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train HRM-Text prompt injection detector")
    parser.add_argument("--test", action="store_true", help="Smoke test on 64 samples")
    parser.add_argument("--lr", type=float, default=2.2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./pi-hrm-text")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--n_layers_H", type=int, default=3)
    parser.add_argument("--n_layers_L", type=int, default=3)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--H_cycles", type=int, default=2)
    parser.add_argument("--L_cycles", type=int, default=3)
    parser.add_argument("--rope_base", type=float, default=10000.0)
    parser.add_argument("--rope_scaling", type=float, default=1.0)
    parser.add_argument("--bp_min", type=int, default=2)
    parser.add_argument("--bp_max", type=int, default=5)
    parser.add_argument("--push_to_hub", type=str, default="av-codes/prompt-injection-hrm-text")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_cache", type=str, default=None,
                        help="Cache dir for dataset download")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 = use epochs)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume from")
    args = parser.parse_args()

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"🖥️  Hardware: {'GPU' if use_cuda else 'CPU'}")
    if use_cuda:
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"📐 HRM-Text(H={args.n_layers_H}, L={args.n_layers_L}, "
          f"d={args.hidden_size}, H_cycles={args.H_cycles}, L_cycles={args.L_cycles})")
    print(f"📏 Max context: {args.max_length:,} tokens")
    print(f"🔄 BP warmup: {args.bp_min}→{args.bp_max} steps")

    # ── Load dataset ──────────────────────────────────────────────────────
    print("\n📦 Loading Bordair multimodal dataset...")
    # For test mode, load a small subset directly
    if args.test:
        # Small test with subset
        all_parts = []
        # Try loading just a few files for testing
        test_path = args.data_cache or "/tmp/bordair_test"
        if not os.path.exists(test_path):
            snapshot_download(
                repo_id="Bordair/bordair-multimodal",
                repo_type="dataset",
                cache_dir=args.data_cache,
                local_dir=test_path,
                local_dir_use_symlinks=False,
                allow_patterns=["benign/text_only.json", "benign/multimodal_text_image.json",
                                "payloads/text_image/text_image_001.json"],
            )
        for f in glob.glob(f"{test_path}/**/*.json", recursive=True):
            fname = os.path.basename(f)
            if fname in ("summary.json", "_pool.json"):
                continue
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("expected_detection") is not None:
                        text_parts = [item.get("text", "")]
                        for k in ("image_content", "document_content", "audio_content"):
                            if item.get(k):
                                text_parts.append(item[k])
                        all_parts.append({
                            "text": "\n".join(text_parts),
                            "label": 1 if item["expected_detection"] else 0,
                        })
        merged = Dataset.from_list(all_parts)
        print(f"   Test mode: {len(merged)} samples loaded")
    else:
        merged = load_bordair_multimodal(cache_dir=args.data_cache)

    # ── Stratified 90/10 split ────────────────────────────────────────────
    # Cast label to ClassLabel first
    merged = merged.cast_column("label", hf_datasets.ClassLabel(names=["safe", "injection"]))

    if args.test:
        train_dataset = merged.select(range(min(64, len(merged))))
        eval_dataset = merged.select(range(min(32, len(merged))))
    else:
        split = merged.train_test_split(
            test_size=0.05, seed=args.seed, stratify_by_column="label",
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]

    print(f"\n✅ Dataset: {len(merged)} total → {len(train_dataset)} train, {len(eval_dataset)} eval")
    train_dist = Counter(train_dataset["label"])
    eval_dist = Counter(eval_dataset["label"])
    print(f"   Train label dist: {dict(train_dist)}")
    print(f"   Eval label dist: {dict(eval_dist)}")

    # ── Log token length statistics for context planning ──────────────────
    train_lengths = [len(t.encode("utf-8", errors="replace")) for t in train_dataset["text"]]
    print(f"   Train text length stats: mean={np.mean(train_lengths):.0f}, "
          f"median={np.median(train_lengths):.0f}, "
          f"p95={np.percentile(train_lengths, 95):.0f}, "
          f"max={max(train_lengths):,}")

    # ── Build model ───────────────────────────────────────────────────────
    model = HrmTextClassifier(
        vocab_size=256,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        n_layers_H=args.n_layers_H,
        n_layers_L=args.n_layers_L,
        intermediate_size=args.intermediate_size,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        max_seq_len=args.max_length,
        rope_base=args.rope_base,
        rope_scaling_factor=args.rope_scaling,
        num_classes=2,
        bp_min_steps=args.bp_min,
        bp_max_steps=args.bp_max,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    param_count = count_params(model)
    print(f"\n🧮 Model parameters: {param_count:,}")
    if args.gradient_checkpointing:
        print("   Gradient checkpointing: enabled")
    if not args.test:
        assert 15_000_000 <= param_count <= 55_000_000, \
            f"Param count {param_count:,} outside target range [15M, 55M]"

    if use_cuda:
        model = model.cuda()

    # ── Metrics ───────────────────────────────────────────────────────────
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "precision": precision.compute(predictions=preds, references=labels, average="binary")["precision"],
            "recall": recall.compute(predictions=preds, references=labels, average="binary")["recall"],
            "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
        }

    # ── Estimate total training steps for BP warmup ───────────────────────
    steps_per_epoch = max(1, len(train_dataset) // args.batch_size)
    total_training_steps = steps_per_epoch * args.epochs
    print(f"📊 Steps per epoch: {steps_per_epoch}, total: {total_training_steps}")

    # ── Training args ─────────────────────────────────────────────────────
    run_name = f"hrm-text-pi_d{args.hidden_size}_lr{args.lr}_ep{args.epochs}_bs{args.batch_size}"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        report_to="none",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=min(args.batch_size * 2, 16),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        weight_decay=0.1,
        warmup_steps=2000 if not args.test else 0,
        lr_scheduler_type="constant_with_warmup",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=5 if args.test else 20,
        disable_tqdm=False if args.test else True,
        fp16=False,
        bf16=use_cuda,
        push_to_hub=True,
        hub_model_id=args.push_to_hub,
        hub_strategy="every_save",
        use_cpu=not use_cuda,
        dataloader_num_workers=4,
        seed=args.seed,
        adam_beta2=0.95,
        save_only_model=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        gradient_checkpointing=False,
    )

    # ── Data collator ─────────────────────────────────────────────────────
    def collate_fn(batch):
        return collate_hrm_text(batch, max_length=args.max_length)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = HrmTextTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        total_training_steps=total_training_steps,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n🚀 Training...")
    train_start = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_elapsed = time.time() - train_start
    print(f"✅ Training complete! ({train_elapsed:.1f}s)")
    print(f"   Best checkpoint: {trainer.state.best_model_checkpoint}")

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\n📊 Evaluating on eval set...")
    eval_metrics = trainer.evaluate(eval_dataset)
    print(f"   Eval metrics: {json.dumps(eval_metrics, indent=2)}")

    os.makedirs(args.output_dir, exist_ok=True)
    eval_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(eval_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # ── Save model locally ────────────────────────────────────────────────
    best_model_path = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    model_path = os.path.join(best_model_path, "hrm_text_model.pt")

    # Try loading best checkpoint first
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt and os.path.isdir(best_ckpt):
        print(f"\n💾 Loading best checkpoint from {best_ckpt}")
        # The checkpoint is saved by trainer; load state dict from it
        # The model in trainer might have DDP wrappers, get unwrapped
        best_model = trainer.model
        if hasattr(best_model, "module"):
            best_model = best_model.module
        torch.save(best_model.state_dict(), model_path)
    else:
        # Save final model
        best_model = trainer.model
        if hasattr(best_model, "module"):
            best_model = best_model.module
        torch.save(best_model.state_dict(), model_path)
        print(f"\n💾 Saved final model weights to {model_path}")

    # Save config
    config = {
        "architecture": "HRM-Text (classification)",
        "reference": "sapientinc/HRM-Text, arXiv:2506.21734",
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "n_layers_H": args.n_layers_H,
        "n_layers_L": args.n_layers_L,
        "intermediate_size": args.intermediate_size,
        "H_cycles": args.H_cycles,
        "L_cycles": args.L_cycles,
        "max_seq_len": args.max_length,
        "rope_base": args.rope_base,
        "rope_scaling": args.rope_scaling,
        "bp_min_steps": args.bp_min,
        "bp_max_steps": args.bp_max,
        "vocab_size": 256,
        "param_count": param_count,
        "id2label": {0: "safe", 1: "injection"},
        "label2id": {"safe": 0, "injection": 1},
        "training": {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "weight_decay": 0.1,
            "scheduler": "constant_with_warmup",
            "warmup_steps": 2000 if not args.test else 0,
            "adam_beta2": 0.95,
            "precision": "bf16",
        },
    }
    with open(os.path.join(best_model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"   Saved config to {best_model_path}/config.json")

    # ── Push to Hub ───────────────────────────────────────────────────────
    if args.push_to_hub:
        hub_model_id = args.push_to_hub
        api = HfApi(token=args.hub_token)

        print(f"\n☁️  Pushing to Hub: {hub_model_id}")

        # Create repo if needed
        try:
            api.create_repo(repo_id=hub_model_id, repo_type="model", private=False, exist_ok=True)
            print(f"   Repo ready: {hub_model_id}")
        except Exception as e:
            print(f"   ⚠️  Could not create repo: {e}")

        # Upload model weights
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="pytorch_model.bin",
            repo_id=hub_model_id,
            repo_type="model",
            commit_message=f"HRM-Text prompt injection detector — F1={eval_metrics.get('eval_f1', 0):.4f}",
        )

        # Upload config
        api.upload_file(
            path_or_fileobj=os.path.join(best_model_path, "config.json"),
            path_in_repo="config.json",
            repo_id=hub_model_id,
            repo_type="model",
            commit_message="Add model config",
        )

        # Upload metrics
        api.upload_file(
            path_or_fileobj=eval_path,
            path_in_repo="eval_metrics.json",
            repo_id=hub_model_id,
            repo_type="model",
            commit_message="Add evaluation metrics",
        )

        # Upload the training script
        script_path = os.path.abspath(__file__) if "__file__" in dir() else None
        if script_path and os.path.exists(script_path):
            api.upload_file(
                path_or_fileobj=script_path,
                path_in_repo="train_hrm_text_pi.py",
                repo_id=hub_model_id,
                repo_type="model",
                commit_message="Add training script",
            )

        # Upload a README
        readme = f"""---
license: mit
tags:
  - prompt-injection
  - hrm-text
  - hierarchical-reasoning-model
  - bordair-multimodal
  - security
---

# HRM-Text Prompt Injection Detector

**Parameters:** {param_count:,}  
**Architecture:** HRM-Text (classification port) | d={args.hidden_size}, H={args.n_layers_H}, L={args.n_layers_L}, cycles={args.H_cycles}×{args.L_cycles}  
**Context window:** {args.max_length:,} tokens (NTK-scaled RoPE)  
**Training data:** Bordair/bordair-multimodal (503K samples, balanced 1:1)  

Evaluation on stratified 10% holdout:

| Metric | Value |
|--------|-------|
| Accuracy | {eval_metrics.get('eval_accuracy', 0):.4f} |
| Precision | {eval_metrics.get('eval_precision', 0):.4f} |
| Recall | {eval_metrics.get('eval_recall', 0):.4f} |
| F1 | {eval_metrics.get('eval_f1', 0):.4f} |

## Architecture

HRM-Text (arXiv:2506.21734) with a classification head. The model uses a recurrent cascade of two transformer modules (H and L) that exchange information across cycles:

- **L module** ({args.n_layers_L} layers, low-level): processes detailed token patterns
- **H module** ({args.n_layers_H} layers, high-level): integrates across cycles  
- **Recurrence**: {args.L_cycles} L-steps per H-cycle, {args.H_cycles} H-cycles total = {args.H_cycles * args.L_cycles} recurrent passes
- **Classification**: last-token pooling + LayerNorm + Linear(2)

The byte-level tokenizer (vocab 256) handles any text encoding. RoPE uses NTK-aware scaling (θ={args.rope_base}, factor={args.rope_scaling}) for {args.max_length:,}-token context.

## Usage
```python
import torch
from train_hrm_text_pi import HrmTextClassifier

model = HrmTextClassifier(
    hidden_size={args.hidden_size},
    num_heads={args.num_heads},
    head_dim={args.head_dim},
    n_layers_H={args.n_layers_H},
    n_layers_L={args.n_layers_L},
)
state_dict = torch.load("pytorch_model.bin", map_location="cpu")
# Remove DDP wrapper keys if present
state_dict = {{k.replace('module.', ''): v for k, v in state_dict.items()}}
model.load_state_dict(state_dict)
model.eval()

def detect(text, max_length=131072):
    byte_ids = list(text.encode("utf-8", errors="replace")[:max_length])
    input_ids = torch.tensor([byte_ids])
    attention_mask = torch.ones_like(input_ids)
    logits = model.inference(input_ids, attention_mask)
    pred = logits.argmax(-1).item()  # 0=safe, 1=injection
    return pred
```
"""
        api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=hub_model_id,
            repo_type="model",
            commit_message="Add README",
        )

        print(f"✅ https://huggingface.co/{hub_model_id}")

    print("\n✅ Done!")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()