#!/usr/bin/env python3
"""
train_hrm_pi.py — Train an HRM (Hierarchical Reasoning Model) classifier
for prompt injection detection on the merged 5-dataset corpus.

Architecture follows arXiv:2506.21734:
  - Input network  f_I: byte-level embedding + position encoding + pooling
  - Low-level      f_L: 2-layer MLP (fast, frequent updates)
  - High-level     f_H: 2-layer MLP (slow, per-cycle updates)
  - Output network f_O: projection to 2 classes

Approximate 1-step gradient + deep supervision.
Target: ~26.5M parameters.
"""
import argparse
import json
import math
import os
from collections import Counter

import datasets as hf_datasets
import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ── Datasets (mirrors train_pi_v2.py) ────────────────────────────────────────
DATASETS = {
    "Shomi28/prompt-injection-dataset": {
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "text_col": "text",
        "label_col": "label",
    },
    "deepset/prompt-injections": {
        "splits": {"train": "train", "test": "test"},
        "text_col": "text",
        "label_col": "label",
    },
    "hendzh/PromptShield": {
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "text_col": "prompt",
        "label_col": "label",
    },
    "watchdogsrox/Mirror-Prompt-Injection-Dataset": {
        "splits": {"train": "train"},
        "text_col": "text",
        "label_col": "label",
    },
    "Antijection/prompt-injection-dataset-v1": {
        "splits": {"train": "train"},
        "text_col": "prompt",
        "label_col": "label",
    },
}


# ── Byte-level tokenizer ─────────────────────────────────────────────────────
class ByteTokenizer:
    """Simple byte-level tokenizer: encodes strings as byte IDs [0-255]."""

    def __init__(self, max_length=256):
        self.max_length = max_length
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<pad>"
        self.vocab_size = 256

    def __call__(self, text_batch, truncation=True, max_length=None, padding=False):
        max_len = max_length or self.max_length
        input_ids = []
        for text in text_batch:
            if isinstance(text, str):
                byte_ids = list(text.encode("utf-8", errors="replace"))
            else:
                byte_ids = []
            if truncation:
                byte_ids = byte_ids[:max_len]
            # Pad to max_len if requested
            if padding:
                pad_len = max_len - len(byte_ids)
                byte_ids = byte_ids + [self.pad_token_id] * pad_len
            input_ids.append(byte_ids)

        # For batch processing, pad to the longest in batch
        if not padding:
            max_in_batch = max(len(ids) for ids in input_ids)
            for ids in input_ids:
                ids.extend([self.pad_token_id] * (max_in_batch - len(ids)))

        attention_mask = [[1] * len(ids) for ids in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def __len__(self):
        return self.vocab_size


# ── HRM Architecture ─────────────────────────────────────────────────────────
class HRMClassifier(nn.Module):
    """
    Hierarchical Reasoning Model for binary text classification.

    Components (per arXiv:2506.21734):
      f_I — Token embedding + position encoding → mean pooling → x̃
      f_L — Low-level recurrent MLP (2-layer, SiLU)
      f_H — High-level recurrent MLP (2-layer, SiLU)
      f_O — Output head: LayerNorm → Linear(2)

    Forward pass:
      1. Encode text → x̃ (average of token + position embeddings)
      2. NT steps: zL ← f_L(zL, zH, x̃); zH ← f_H(zH, zL) every T steps
      3. Predict: ŷ = f_O(zH)

    Training uses the approximate 1-step gradient (only last step trains)
    with deep supervision every n_supervision cycles.
    """

    def __init__(
        self,
        vocab_size=256,
        d_emb=512,
        d_state=2048,
        max_length=256,
        N_cycles=3,
        T_steps=4,
        n_supervision=3,
    ):
        super().__init__()
        self.d_emb = d_emb
        self.d_state = d_state
        self.N_cycles = N_cycles
        self.T_steps = T_steps
        self.n_supervision = min(n_supervision, N_cycles)
        self.max_length = max_length

        # ── f_I: Input network ───────────────────────────────────────────
        self.token_embedding = nn.Embedding(vocab_size, d_emb, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, d_emb) * 0.02)
        self.input_norm = nn.LayerNorm(d_emb)

        # ── f_L: Low-level recurrent module ──────────────────────────────
        # Input: zL(d_state) + zH(d_state) + x̃(d_emb)  →  zL(d_state)
        self.f_L = nn.Sequential(
            nn.Linear(d_state + d_state + d_emb, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state),
            nn.SiLU(),
        )

        # ── f_H: High-level recurrent module ─────────────────────────────
        # Input: zH(d_state) + zL(d_state)  →  zH(d_state)
        self.f_H = nn.Sequential(
            nn.Linear(d_state + d_state, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state),
            nn.SiLU(),
        )

        # ── f_O: Output network ──────────────────────────────────────────
        self.f_O = nn.Sequential(
            nn.LayerNorm(d_state),
            nn.Linear(d_state, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.f_L, self.f_H, self.f_O]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    def encode_text(self, input_ids, attention_mask):
        """Encode text tokens → single vector x̃ via mean pooling."""
        B, L = input_ids.shape
        emb = self.token_embedding(input_ids)  # (B, L, d_emb)
        pos = self.pos_embedding[:, :L, :]  # (1, L, d_emb)
        emb = emb + pos
        emb = self.input_norm(emb)
        # Mean pooling over non-padded tokens
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        x_tilde = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, d_emb)
        return x_tilde

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Full forward pass with deep supervision.

        Uses the approximate 1-step gradient (arXiv:2506.21734 §2):
        - N*T-1 steps with torch.no_grad()
        - Final step with full gradient tracking
        - Deep supervision: logits extracted after each cycle,
          gradients detached between cycles.

        Returns dict with logits and (if labels given) loss.
        """
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        x_tilde = self.encode_text(input_ids, attention_mask)  # (B, d_emb)
        B = x_tilde.size(0)
        device = x_tilde.device

        # Initialize hidden states
        zL = torch.zeros(B, self.d_state, device=device)
        zH = torch.zeros(B, self.d_state, device=device)

        supervision_gap = max(1, self.N_cycles // max(1, self.n_supervision))
        losses = []
        final_logits = None

        for cycle in range(self.N_cycles):
            # ── Low-level: T-1 steps with no_grad ──
            for _ in range(self.T_steps - 1):
                with torch.no_grad():
                    zL = self.f_L(torch.cat([zL, zH, x_tilde], dim=-1))

            # ── Last L-step with gradient tracking ──
            zL = self.f_L(torch.cat([zL, zH, x_tilde], dim=-1))
            # ── H-module update with gradient tracking ──
            zH = self.f_H(torch.cat([zH, zL], dim=-1))

            # ── Deep supervision checkpoint ──
            if (cycle + 1) % supervision_gap == 0 or cycle == self.N_cycles - 1:
                logits = self.f_O(zH)
                final_logits = logits
                if labels is not None:
                    loss = F.cross_entropy(logits, labels)
                    losses.append(loss)

            # Detach hidden states for next cycle (1-step approx)
            zL = zL.detach()
            zH = zH.detach()

        total_loss = sum(losses) / len(losses) if losses else None
        return {"logits": final_logits, "loss": total_loss}

    @torch.no_grad()
    def inference(self, input_ids, attention_mask=None):
        """Inference-only forward pass (no gradients, no deep supervision loss).

        Runs all N*T steps without gradient tracking, closely matching
        the full forward dynamics.
        """
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        x_tilde = self.encode_text(input_ids, attention_mask)
        B = x_tilde.size(0)
        device = x_tilde.device

        zL = torch.zeros(B, self.d_state, device=device)
        zH = torch.zeros(B, self.d_state, device=device)

        for _ in range(self.N_cycles):
            for _ in range(self.T_steps):
                zL = self.f_L(torch.cat([zL, zH, x_tilde], dim=-1))
            zH = self.f_H(torch.cat([zH, zL], dim=-1))

        logits = self.f_O(zH)
        return logits


# ── Data helpers ─────────────────────────────────────────────────────────────


def normalize_label(ex, label_col):
    """Convert label to int64 0/1."""
    val = ex[label_col]
    if isinstance(val, str):
        return {label_col: 1 if val.lower() in ("malicious", "injection", "yes", "1") else 0}
    return {label_col: int(val)}


def collate_bytes(batch):
    """Collate function for byte-level tokenized examples."""
    texts = [ex["text"] for ex in batch]
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)

    # Encode to byte IDs in one pass
    all_ids = []
    for t in texts:
        ids = list(t.encode("utf-8", errors="replace")[:256])
        all_ids.append(ids)

    max_len = max(len(ids) for ids in all_ids)
    all_ids_padded = []
    attention_masks = []
    for ids in all_ids:
        length = len(ids)
        padded = ids + [0] * (max_len - length)
        mask = [1] * length + [0] * (max_len - length)
        all_ids_padded.append(padded)
        attention_masks.append(mask)

    return {
        "input_ids": torch.tensor(all_ids_padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": labels,
    }


# ── Custom Trainer ───────────────────────────────────────────────────────────


class HRMTrainer(Trainer):
    """Trainer subclass that handles HRM's custom forward signature."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        labels = inputs.pop("labels") if "labels" in inputs else None
        with torch.no_grad():
            logits = model.inference(**inputs)
        if prediction_loss_only:
            return (None, None, labels)
        return (None, logits, labels)


# ── Main ─────────────────────────────────────────────────────────────────────


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Smoke test on 64 samples")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./pi-hrm")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--d_emb", type=int, default=512)
    parser.add_argument("--d_state", type=int, default=2048)
    parser.add_argument("--N_cycles", type=int, default=3)
    parser.add_argument("--T_steps", type=int, default=4)
    parser.add_argument("--n_supervision", type=int, default=3)
    parser.add_argument("--push_to_hub", type=str, default="av-codes/prompt-injection-hrm")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.cpu
    print(f"🖥️  Hardware: {'GPU' if use_cuda else 'CPU'}")
    print(f"📐 HRM(d_emb={args.d_emb}, d_state={args.d_state}, "
          f"N={args.N_cycles}, T={args.T_steps})")

    # ── Load & merge training datasets ──────────────────────────────────────
    all_parts = []
    print("\n📦 Loading datasets...")
    for ds_path, cfg in DATASETS.items():
        try:
            ds = load_dataset(ds_path)
        except Exception as e:
            print(f"   ⚠️  {ds_path}: failed to load — {e}")
            continue

        text_col = cfg["text_col"]
        label_col = cfg["label_col"]

        for split_key in ("train", "test", "val"):
            if split_key not in cfg["splits"]:
                continue
            split_name = cfg["splits"][split_key]
            part = ds[split_name]
            part = part.map(lambda ex: normalize_label(ex, label_col))
            part = part.cast_column(label_col, hf_datasets.Value("int64"))
            if text_col != "text":
                part = part.rename_column(text_col, "text")
            if label_col != "label":
                part = part.rename_column(label_col, "label")
            part = part.select_columns(["text", "label"])
            all_parts.append(part)

        cumulative = sum(len(p) for p in all_parts)
        print(f"   ✓ {ds_path}: {cumulative} cumulative")

    merged = concatenate_datasets(all_parts).flatten_indices()

    # Stratified 90/10 train/eval split
    merged = merged.cast_column("label", hf_datasets.ClassLabel(names=["safe", "injection"]))
    split = merged.train_test_split(test_size=0.1, seed=args.seed, stratify_by_column="label")
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"\n✅ Merged: {len(merged)} total → {len(train_dataset)} train, {len(eval_dataset)} eval")
    train_dist = Counter(train_dataset["label"])
    eval_dist = Counter(eval_dataset["label"])
    print(f"   Train label dist: {dict(train_dist)}")
    print(f"   Eval label dist: {dict(eval_dist)}")

    # ── Build model ─────────────────────────────────────────────────────────
    model = HRMClassifier(
        vocab_size=256,
        d_emb=args.d_emb,
        d_state=args.d_state,
        max_length=args.max_length,
        N_cycles=args.N_cycles,
        T_steps=args.T_steps,
        n_supervision=args.n_supervision,
    )
    param_count = count_params(model)
    print(f"\n🧮 Model parameters: {param_count:,}")
    assert 20_000_000 <= param_count <= 27_000_000, \
        f"Param count {param_count:,} outside target range [20M, 27M]"

    if use_cuda:
        model = model.cuda()

    # ── Smoke test ──────────────────────────────────────────────────────────
    if args.test:
        train_dataset = train_dataset.select(range(64))
        eval_dataset = eval_dataset.select(range(32))

    # ── Metrics ─────────────────────────────────────────────────────────────
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

    # ── Training args ───────────────────────────────────────────────────────
    run_name = f"hrm-pi_d{args.d_state}_lr{args.lr}_ep{args.epochs}_bs{args.batch_size}"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        report_to="none",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=50 if not args.test else 0,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=5 if args.test else 20,
        disable_tqdm=False if args.test else True,
        fp16=use_cuda,
        bf16=False,
        push_to_hub=False,
        hub_model_id=None,
        use_cpu=not use_cuda,
        dataloader_num_workers=0,
        seed=args.seed,
        save_only_model=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = HRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_bytes,
        compute_metrics=compute_metrics,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    print("\n🚀 Training...")
    trainer.train()
    print("✅ Training complete!")

    # ── Final evaluation ────────────────────────────────────────────────────
    print("\n📊 Evaluating on eval set...")
    eval_metrics = trainer.evaluate(eval_dataset)
    print(f"   Eval metrics: {json.dumps(eval_metrics, indent=2)}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # ── Save best checkpoint locally ────────────────────────────────────────
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt and os.path.isdir(best_ckpt):
        print(f"\n💾 Best checkpoint: {best_ckpt}")
        best_model_path = os.path.join(args.output_dir, "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        model_path = os.path.join(best_model_path, "hrm_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"   Saved model weights to {model_path}")
    else:
        # Save final model
        best_model_path = os.path.join(args.output_dir, "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        model_path = os.path.join(best_model_path, "hrm_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"   No best checkpoint found, saved final weights to {model_path}")

    # Save model config
    config = {
        "d_emb": args.d_emb,
        "d_state": args.d_state,
        "max_length": args.max_length,
        "N_cycles": args.N_cycles,
        "T_steps": args.T_steps,
        "n_supervision": args.n_supervision,
        "param_count": param_count,
        "vocab_size": 256,
        "id2label": {0: "safe", 1: "injection"},
        "label2id": {"safe": 0, "injection": 1},
    }
    with open(os.path.join(best_model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"   Saved config to {best_model_path}/config.json")

    # ── Push to Hub ─────────────────────────────────────────────────────────
    if args.push_to_hub:
        from huggingface_hub import HfApi

        hub_model_id = args.push_to_hub
        api = HfApi(token=args.hub_token)

        print(f"\n☁️  Pushing to Hub: {hub_model_id}")

        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=hub_model_id, repo_type="model", private=False, exist_ok=True)
            print(f"   Repo ready/created: {hub_model_id}")
        except Exception as e:
            print(f"   ⚠️  Could not create repo: {e}")
            # Try to proceed anyway

        # Upload model weights
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="hrm_model.pt",
            repo_id=hub_model_id,
            repo_type="model",
            commit_message=f"HRM prompt injection detector — F1={eval_metrics.get('eval_f1', 0):.4f}",
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
            path_or_fileobj=os.path.join(args.output_dir, "eval_metrics.json"),
            path_in_repo="eval_metrics.json",
            repo_id=hub_model_id,
            repo_type="model",
            commit_message="Add evaluation metrics",
        )

        # Upload a README
        readme = f"""---
license: mit
tags:
  - prompt-injection
  - hrm
  - hierarchical-reasoning-model
---

# HRM Prompt Injection Detector

**Parameters:** {param_count:,}  
**Architecture:** HRM (arXiv:2506.21734) | d_emb={args.d_emb}, d_state={args.d_state}, N={args.N_cycles}, T={args.T_steps}

Trained on merged 5-dataset prompt injection corpus with stratified 90/10 split.

## Evaluation
| Metric | Value |
|--------|-------|
| Accuracy | {eval_metrics.get('eval_accuracy', 0):.4f} |
| Precision | {eval_metrics.get('eval_precision', 0):.4f} |
| Recall | {eval_metrics.get('eval_recall', 0):.4f} |
| F1 | {eval_metrics.get('eval_f1', 0):.4f} |

## Usage
```python
import torch
from train_hrm_pi import HRMClassifier, ByteTokenizer

model = HRMClassifier(d_emb={args.d_emb}, d_state={args.d_state})
model.load_state_dict(torch.load("hrm_model.pt", map_location="cpu"))
model.eval()

tokenizer = ByteTokenizer(max_length={args.max_length})
tokens = tokenizer(["Your prompt here"])
logits = model.inference(tokens["input_ids"], tokens["attention_mask"])
pred = logits.argmax(-1).item()  # 0=safe, 1=injection
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