#!/usr/bin/env python3
"""
Train a tiny DistilBERT model for prompt injection detection.

Combines Shomi28/prompt-injection-dataset (1K) + deepset/prompt-injections (546)
for a total of ~1.5K training samples.

Usage:
    python train_pi_detector.py              # full train on CPU
    python train_pi_detector.py --test       # quick smoke test on 16 samples
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


class EvalAndPushCallback(TrainerCallback):
    """Runs final test evaluation and pushes model to Hub after training."""

    def __init__(self, test_dataset, data_collator, accuracy, precision, recall, f1, batch_size, hub_model_id, tokenizer):
        self.test_dataset = test_dataset
        self.data_collator = data_collator
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.batch_size = batch_size
        self.hub_model_id = hub_model_id
        self.tokenizer = tokenizer

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        print("\n📊 Evaluating on test set...")
        model.eval()
        all_preds, all_labels = [], []
        # Remove non-tensor columns for direct DataLoader usage
        # Remove non-tensor columns for direct DataLoader usage, rename label→labels
        test_ds = self.test_dataset.rename_column("label", "labels")
        test_ds = test_ds.remove_columns(
            [c for c in test_ds.column_names if c not in ["input_ids", "attention_mask", "labels"]]
        )
        dl = DataLoader(
            test_ds,
            batch_size=self.batch_size * 2,
            collate_fn=self.data_collator,
            shuffle=False,
        )
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = model(**batch)
                preds = outputs.logits.argmax(-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_metrics = {
            "accuracy": self.accuracy.compute(predictions=list(all_preds), references=list(all_labels))["accuracy"],
            "precision": self.precision.compute(predictions=list(all_preds), references=list(all_labels), average="binary")["precision"],
            "recall": self.recall.compute(predictions=list(all_preds), references=list(all_labels), average="binary")["recall"],
            "f1": self.f1.compute(predictions=list(all_preds), references=list(all_labels), average="binary")["f1"],
        } if len(all_preds) > 0 else {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        print("\n🎯 Test Results:")
        for k, v in sorted(test_metrics.items()):
            print(f"   {k}: {v:.4f}")

        # Log via trackio (may have been shut down after training completes)
        try:
            import trackio
            trackio.log({f"test/{k}": v for k, v in test_metrics.items()})
            trackio.alert("Training complete", f"Test F1={test_metrics.get('f1', 0):.4f}", level="info")
        except Exception:
            # trackio may already be finalized; log to stdout instead
            print(f"[trackio skipped — already finalized] test metrics: {test_metrics}")

        # Save metrics
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

        # Push to hub
        if self.hub_model_id:
            print(f"\n☁️  Pushing to Hub: {self.hub_model_id}")
            if hasattr(self, 'trainer_ref') and self.trainer_ref is not None:
                self.trainer_ref.push_to_hub(
                    commit_message=f"Prompt injection detector — F1={test_metrics.get('f1', 0):.4f}"
                )
            self.tokenizer.push_to_hub(self.hub_model_id)
            print(f"✅ Model pushed to https://huggingface.co/{self.hub_model_id}")


def main():
    # ── Parse args ────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Smoke test on 16 samples")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./pi-detector")
    parser.add_argument("--push_to_hub", type=str, default=None, help="HF repo name to push to")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (no CUDA check)")
    args = parser.parse_args()

    set_seed(42)

    use_cuda = torch.cuda.is_available() and not args.cpu
    print(f"🖥️  Hardware: {'GPU' if use_cuda else 'CPU'} (CUDA available: {torch.cuda.is_available()})")

    # ── Load datasets ─────────────────────────────────────────────────────────────
    print("📦 Loading Shomi28/prompt-injection-dataset...")
    ds1 = load_dataset("Shomi28/prompt-injection-dataset")
    print(f"   Train: {len(ds1['train'])}, Val: {len(ds1['validation'])}, Test: {len(ds1['test'])}")

    print("📦 Loading deepset/prompt-injections...")
    ds2 = load_dataset("deepset/prompt-injections")
    print(f"   Train: {len(ds2['train'])}, Test: {len(ds2['test'])}")

    # ── Merge ─────────────────────────────────────────────────────────────────────
    ds1_train = ds1["train"].remove_columns(["label_name"])
    ds1_val = ds1["validation"].remove_columns(["label_name"])
    ds1_test = ds1["test"].remove_columns(["label_name"])

    ds1_train = ds1_train.cast_column("label", hf_datasets.Value("int64"))
    ds1_val = ds1_val.cast_column("label", hf_datasets.Value("int64"))
    ds1_test = ds1_test.cast_column("label", hf_datasets.Value("int64"))

    train_dataset = concatenate_datasets([ds1_train, ds2["train"]])
    test_dataset = concatenate_datasets([ds1_test, ds2["test"]])

    print(f"\n✅ Combined dataset: {len(train_dataset)} train, {len(ds1_val)} val, {len(test_dataset)} test")

    label_counts = [train_dataset[i]["label"] for i in range(min(len(train_dataset), 1000))]
    dist = Counter(label_counts)
    print(f"   Train label dist (first 1K): {dict(dist)}")

    # ── Tokenizer & model ─────────────────────────────────────────────────────────
    print(f"\n🤖 Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "safe", 1: "injection"},
        label2id={"safe": 0, "injection": 1},
    )

    # ── Preprocess ────────────────────────────────────────────────────────────────
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding=False)

    if args.test:
        train_dataset = train_dataset.select(range(16))
        ds1_val = ds1_val.select(range(8))
        test_dataset = test_dataset.select(range(8))

    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_val = ds1_val.map(tokenize_fn, batched=True)
    tokenized_test = test_dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Metrics ───────────────────────────────────────────────────────────────────
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

    # ── Training args ─────────────────────────────────────────────────────────────
    run_name = f"pi-detector-distilbert-lr{args.lr}-ep{args.epochs}-bs{args.batch_size}"
    hub_model_id = args.push_to_hub

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        report_to="trackio",
        project="prompt-injection-detector",
        trackio_space_id=os.environ.get("TRACKIO_SPACE_ID", None),
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
        logging_steps=5 if args.test else 10,
        disable_tqdm=False if args.test else True,
        fp16=use_cuda,
        bf16=False,
        push_to_hub=False,  # We'll push manually in callback
        hub_model_id=hub_model_id,
        use_cpu=not use_cuda,
        dataloader_num_workers=0,
        seed=42,
        save_only_model=True,
    )

    # ── Callbacks ────────────────────────────────────────────────────────────────
    eval_callback = EvalAndPushCallback(
        test_dataset=tokenized_test,
        data_collator=data_collator,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        batch_size=args.batch_size,
        hub_model_id=hub_model_id,
        tokenizer=tokenizer,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[eval_callback],
    )

    # Give callback a reference to trainer
    eval_callback.trainer_ref = trainer

    # ── Train ─────────────────────────────────────────────────────────────────────
    print("\n🚀 Training...")
    trainer.train()
    print("\n✅ Done!")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()