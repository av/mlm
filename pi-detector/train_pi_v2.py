#!/usr/bin/env python3
"""
Prompt Injection Detector v2 — unified training for DistilBERT (Track A) and MiniLM (Track B1).

Merges 5 datasets for ~16K training samples with Mirror-style attack-category balancing.
"""
import argparse
import json
import os
from collections import Counter

import datasets as hf_datasets
import evaluate
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


# ── Datasets ────────────────────────────────────────────────────────────────
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

OVERDEFENSE_DATASETS = {
    "leolee99/NotInject": {
        "splits": {"one": "NotInject_one", "two": "NotInject_two", "three": "NotInject_three"},
        "text_col": "prompt",
    },
}


class OverDefenseCallback(TrainerCallback):
    """Evaluate on over-defense datasets after training."""

    def __init__(self, overdefense_datasets, batch_size, data_collator):
        self.overdefense_datasets = overdefense_datasets
        self.batch_size = batch_size
        self.data_collator = data_collator

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        device = model.device

        print("\n📊 Evaluating on over-defense datasets...")
        total_fp = 0
        total_samples = 0
        for ds_name, (_, tokenized_ds) in self.overdefense_datasets.items():
            td = tokenized_ds.remove_columns(
                [c for c in tokenized_ds.column_names if c not in ["input_ids", "attention_mask"]]
            )
            dl = DataLoader(td, batch_size=self.batch_size, collate_fn=self.data_collator, shuffle=False)
            fp = 0
            model.eval()
            with torch.no_grad():
                for batch in dl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    preds = outputs.logits.argmax(-1)
                    fp += preds.cpu().sum().item()
            print(f"   {ds_name}: {fp}/{len(td)} false positives ({fp/len(td)*100:.1f}%)")
            total_fp += fp
            total_samples += len(td)
        if total_samples > 0:
            print(f"   Total: {total_fp}/{total_samples} FP ({total_fp/total_samples*100:.1f}%)\n")


def normalize_label(ds, label_col):
    """Convert label to int64 0/1."""
    def _norm(ex):
        val = ex[label_col]
        if isinstance(val, str):
            return {label_col: 1 if val.lower() in ("malicious", "injection", "yes", "1") else 0}
        return {label_col: int(val)}
    return ds.map(_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Smoke test on 32 samples")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="distilbert/distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./pi-detector-v2")
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    set_seed(42)
    use_cuda = torch.cuda.is_available() and not args.cpu
    print(f"🖥️  Hardware: {'GPU' if use_cuda else 'CPU'}")
    print(f"📐 Model: {args.model_name}")

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
            part = normalize_label(part, label_col)
            part = part.cast_column(label_col, hf_datasets.Value("int64"))
            if text_col != "text":
                part = part.rename_column(text_col, "text")
            if label_col != "label":
                part = part.rename_column(label_col, "label")
            part = part.select_columns(["text", "label"])
            all_parts.append(part)

        total = sum(len(p) for p in all_parts) - sum(len(p) for p in all_parts[:-1]) if all_parts else 0
        print(f"   ✓ {ds_path}: {sum(len(p) for p in all_parts)} cumulative")

    merged = concatenate_datasets(all_parts).flatten_indices()

    # Stratified 90/10 train/eval split from merged data
    merged = merged.cast_column("label", hf_datasets.ClassLabel(names=["safe", "injection"]))
    split = merged.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
    train_dataset = split["train"]
    test_dataset = split["test"]

    print(f"\n✅ Merged: {len(merged)} total → {len(train_dataset)} train, {len(test_dataset)} eval")
    dist = Counter(train_dataset["label"])
    print(f"   Train label dist: {dict(dist)}")
    dist = Counter(test_dataset["label"])
    print(f"   Eval label dist: {dict(dist)}")

    # ── Tokenizer & model ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "safe", 1: "injection"},
        label2id={"safe": 0, "injection": 1},
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Preprocess ─────────────────────────────────────────────────────────
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length, padding=False)

    if args.test:
        train_dataset = train_dataset.select(range(32))
        if test_dataset:
            test_dataset = test_dataset.select(range(16))

    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_test = test_dataset.map(tokenize_fn, batched=True) if test_dataset else None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Metrics ────────────────────────────────────────────────────────────
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

    # ── Over-defense datasets ──────────────────────────────────────────────
    overdefense_data = {}
    for ds_path, cfg in OVERDEFENSE_DATASETS.items():
        try:
            ods = load_dataset(ds_path)
            text_col = cfg["text_col"]
            parts = []
            for name, split in cfg["splits"].items():
                part = ods[split].rename_column(text_col, "text")
                part = part.select_columns(["text"])
                parts.append(part)
            combined = concatenate_datasets(parts)
            tokenized_od = combined.map(tokenize_fn, batched=True)
            overdefense_data[ds_path] = (combined, tokenized_od)
            print(f"   📋 Over-defense: {ds_path} — {len(combined)} samples")
        except Exception as e:
            print(f"   ⚠️  Over-defense {ds_path}: {e}")

    # ── Training args ──────────────────────────────────────────────────────
    model_short = args.model_name.split("/")[-1]
    run_name = f"pi-v2_{model_short}_lr{args.lr}_ep{args.epochs}_bs{args.batch_size}"
    hub_model_id = args.push_to_hub

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        report_to="none",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=0 if args.test else 100,
        lr_scheduler_type="linear",
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
        hub_model_id=hub_model_id,
        use_cpu=not use_cuda,
        dataloader_num_workers=0,
        seed=42,
        save_only_model=True,
    )

    # ── Callbacks ──────────────────────────────────────────────────────────
    callbacks = []
    if overdefense_data:
        callbacks.append(OverDefenseCallback(overdefense_data, args.batch_size, data_collator))

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n🚀 Training...")
    trainer.train()
    print("✅ Training complete!")

    # ── Final test evaluation ──────────────────────────────────────────────
    if tokenized_test:
        print("\n📊 Evaluating on test set...")
        test_metrics = trainer.evaluate(tokenized_test)
        print(f"   Test metrics: {json.dumps(test_metrics, indent=2)}")

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

    # ── Push to Hub ────────────────────────────────────────────────────────
    if hub_model_id:
        print(f"\n☁️  Pushing to Hub: {hub_model_id}")
        trainer.push_to_hub(commit_message=f"Prompt injection v2 — F1={test_metrics.get('eval_f1', 0):.4f}" if tokenized_test else "Prompt injection v2")
        tokenizer.push_to_hub(hub_model_id)
        print(f"✅ https://huggingface.co/{hub_model_id}")

    print("\n✅ Done!")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()