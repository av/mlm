#!/usr/bin/env python3
"""
TRM-Att (Tiny Recursive Model with Attention) for text classification.

Based on:
  "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)
  lucidrains/tiny-recursive-model v0.0.15

Architecture: dim=512, depth=6, heads=8 → ~19M params
Trains on prompt injection detection data with a custom recursive loop.
"""

import argparse
import json
import math
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tiny_recursive_model import TinyRecursiveModel
from x_transformers import Encoder

import datasets as hf_datasets
from datasets import concatenate_datasets, load_dataset

# ── Byte-level tokenizer (vocab_size=256, keeps embedding tiny) ────────────

class ByteTokenizer:
    """Simple byte-level tokenizer: text → bytes → ints."""
    def __init__(self, max_length=256):
        self.vocab_size = 256
        self.pad_token_id = 0
        self.max_length = max_length

    def encode(self, text):
        """Encode text to list of ints (bytes), truncated to max_length."""
        if isinstance(text, str):
            text = text.encode("utf-8", errors="replace")
        return [b for b in text[:self.max_length]]

    def encode_batch(self, texts):
        """Encode and pad a batch of texts."""
        batch = []
        for t in texts:
            ids = self.encode(t)
            if len(ids) < self.max_length:
                ids = ids + [self.pad_token_id] * (self.max_length - len(ids))
            else:
                ids = ids[:self.max_length]
            batch.append(ids)
        return torch.tensor(batch, dtype=torch.long)


# ── Dataset wrapper ────────────────────────────────────────────────────────

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx])
        if len(ids) < self.max_length:
            ids = ids + [self.tokenizer.pad_token_id] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": self.labels[idx],
        }


def pad_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


# ── TRM Classifier wrapper ────────────────────────────────────────────────

class TRMForClassification(nn.Module):
    """Wraps TinyRecursiveModel for text classification.

    Replaces token-prediction head with a classification head.
    Runs T refinement blocks (deep supervision on all blocks).
    """

    def __init__(
        self,
        dim=512,
        depth=6,
        heads=8,
        vocab_size=256,
        num_classes=2,
        num_refinement_blocks=3,
        num_latent_refinements=6,
        num_register_tokens=4,
    ):
        super().__init__()

        self.dim = dim
        self.num_refinement_blocks = num_refinement_blocks
        self.num_register_tokens = num_register_tokens

        # Encoder network (TRM-Att variant)
        network = Encoder(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_glu=True,  # SwiGLU FFN as in paper
        )

        self.trm = TinyRecursiveModel(
            dim=dim,
            num_tokens=vocab_size,
            network=network,
            num_refinement_blocks=num_refinement_blocks,
            num_latent_refinements=num_latent_refinements,
            num_register_tokens=num_register_tokens,
        )

        # Replace prediction head: token_pred → classifier
        self.trm.to_pred = nn.Linear(dim, num_classes, bias=False)

        # Count parameters
        self._param_count = sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, labels=None):
        """Forward pass: single call to TRM (handles deep refinement internally),
        then pool register token for classification."""
        outputs, latents = self.trm.get_initial()  # (dim,) — broadcastable

        # TRM forward: embeds + runs deep refinement + predicts
        # Do NOT pass labels — we handle classification loss ourselves
        outputs, latents, pred, halt_prob = self.trm(
            input_ids, outputs, latents
        )

        # Pool first register token for classification
        logits = pred[:, 0]  # (b, num_classes)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    def get_param_count(self):
        return self._param_count


# ── Training ──────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, scheduler, epoch, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        output = model(input_ids, labels=labels)
        loss = output["loss"]
        logits = output["logits"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * input_ids.shape[0]
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_samples += input_ids.shape[0]

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{total_correct/total_samples:.3f}",
        })

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids, labels=labels)
        loss = output["loss"]
        logits = output["logits"]

        total_loss += loss.item() * input_ids.shape[0]
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    preds = all_preds
    refs = all_labels
    n = len(preds)
    avg_loss = total_loss / n if n > 0 else 0
    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(refs, preds),
        "precision": precision_score(refs, preds, average="binary", zero_division=0),
        "recall": recall_score(refs, preds, average="binary", zero_division=0),
        "f1": f1_score(refs, preds, average="binary", zero_division=0),
    }


# ── Load prompt injection data ────────────────────────────────────────────

def load_pi_data(test=False, max_length=256):
    """Load and merge prompt injection datasets."""
    datasets_config = [
        ("Shomi28/prompt-injection-dataset", "train", "text", "label"),
        ("deepset/prompt-injections", "train", "text", "label"),
        ("hendzh/PromptShield", "train", "prompt", "label"),
        ("watchdogsrox/Mirror-Prompt-Injection-Dataset", "train", "text", "label"),
        ("Antijection/prompt-injection-dataset-v1", "train", "prompt", "label"),
    ]

    all_texts, all_labels = [], []

    for ds_path, split, text_col, label_col in datasets_config:
        try:
            ds = load_dataset(ds_path, split=split)
        except Exception as e:
            print(f"   ⚠️  {ds_path}: {e}")
            continue

        # Normalize labels
        def norm_label(ex):
            val = ex[label_col]
            if isinstance(val, str):
                return 1 if val.lower() in ("malicious", "injection", "yes", "1") else 0
            return int(val)

        texts = ds[text_col]
        labels = [norm_label(ex) for ex in ds]

        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"   ✓ {ds_path}: {len(texts)} samples")

    if test:
        all_texts = all_texts[:100]
        all_labels = all_labels[:100]

    print(f"\n✅ Total: {len(all_texts)} samples")
    dist = Counter(all_labels)
    print(f"   Label dist: {dict(dist)}")

    tokenizer = ByteTokenizer(max_length=max_length)
    dataset = TextClassificationDataset(all_texts, all_labels, tokenizer, max_length)

    # Split into train/val
    n = len(dataset)
    n_val = int(n * 0.1)
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    return train_ds, val_ds, n_train, n_val


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Quick smoke test")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./trm-pi-detector")
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_refinement_blocks", type=int, default=3)
    parser.add_argument("--num_latent_refinements", type=int, default=6)
    parser.add_argument("--num_register_tokens", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📐 Config: dim={args.dim}, depth={args.depth}, heads={args.heads}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n📦 Loading data...")
    train_ds, val_ds, n_train, n_val = load_pi_data(
        test=args.test, max_length=args.max_length
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=pad_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=pad_collate_fn, num_workers=0,
    )

    # ── Build model ────────────────────────────────────────────────────────
    print("\n🏗️  Building TRM classifier...")
    model = TRMForClassification(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        vocab_size=256,  # byte-level
        num_classes=2,
        num_refinement_blocks=args.num_refinement_blocks,
        num_latent_refinements=args.num_latent_refinements,
        num_register_tokens=args.num_register_tokens,
    ).to(device)

    param_count = model.get_param_count()
    print(f"   Parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Linear warmup + cosine decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(2000, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n🚀 Training ({args.epochs} epochs, {total_steps} steps)...")
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, device
        )

        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["f1"]

        print(
            f"   Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f}, "
            f"acc: {val_metrics['accuracy']:.4f}, "
            f"prec: {val_metrics['precision']:.4f}, "
            f"rec: {val_metrics['recall']:.4f}, "
            f"f1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"   ✨ New best F1: {best_f1:.4f}")

    # ── Final ──────────────────────────────────────────────────────────────
    print(f"\n✅ Best validation F1: {best_f1:.4f}")

    metrics = {"best_f1": best_f1, "params": param_count}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.push_to_hub:
        print(f"\n☁️  Pushing to Hub: {args.push_to_hub} not yet supported for custom models")
        print("   Saving checkpoint locally for manual upload.")

    print("✅ Done!")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()