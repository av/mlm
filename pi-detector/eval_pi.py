#!/usr/bin/env python3
"""
Quick evaluation of trained prompt injection detectors.

Usage:
  python eval_pi.py                        # evaluate all trained models
  python eval_pi.py --model ./pi-detector-v2-distilbert/checkpoint-XX
  python eval_pi.py --hub av-codes/prompt-injection-detector-v2
"""

import argparse
import json
import os
import sys

import evaluate
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATASETS = {
    "PromptShield test": ("hendzh/PromptShield", "test", "prompt", "label"),
    "deepset test": ("deepset/prompt-injections", "test", "text", "label"),
    "Shomi28 test": ("Shomi28/prompt-injection-dataset", "test", "text", "label"),
}

OVERDEFENSE = {
    "NotInject": ("leolee99/NotInject", None, "prompt"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Path to local model checkpoint or hub model ID")
    parser.add_argument("--hub", type=str, default=None,
                        help="Hub model ID (alternative to --model)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save metrics to file")
    args = parser.parse_args()

    if args.hub:
        model_id = args.hub
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif args.model:
        model_id = args.model
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        # Try local checkpoints
        paths = [
            "./pi-detector-v2-distilbert/checkpoint-495",
            "./pi-detector-v2-minilm/checkpoint-495",
            "./pi-detector/checkpoint-495",
        ]
        for p in paths:
            if os.path.exists(p):
                model_id = p
                model = AutoModelForSequenceClassification.from_pretrained(p)
                tokenizer = AutoTokenizer.from_pretrained(p)
                break
        else:
            print("No local model found. Use --model or --hub")
            sys.exit(1)

    print(f"📐 Model: {model_id}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    results = {}

    # ── Test datasets ──────────────────────────────────────────────────────
    for name, (ds_path, split, text_col, label_col) in DATASETS.items():
        try:
            ds = load_dataset(ds_path, split=split)
        except Exception:
            continue

        def norm(ex):
            val = ex[label_col]
            if isinstance(val, str):
                return 1 if val.lower() in ("malicious", "injection", "yes", "1") else 0
            return int(val)

        all_preds, all_refs = [], []
        for i in range(len(ds)):
            ex = ds[i]
            inputs = tokenizer(ex[text_col], return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model(**inputs)
            pred = output.logits.argmax(-1).item()
            ref = norm(ex)
            all_preds.append(pred)
            all_refs.append(ref)

        results[name] = {
            "accuracy": accuracy.compute(predictions=all_preds, references=all_refs)["accuracy"],
            "precision": precision.compute(predictions=all_preds, references=all_refs, average="binary")["precision"],
            "recall": recall.compute(predictions=all_preds, references=all_refs, average="binary")["recall"],
            "f1": f1.compute(predictions=all_preds, references=all_refs, average="binary")["f1"],
            "samples": len(all_preds),
        }

    # ── Over-defense datasets ──────────────────────────────────────────────
    for name, (ds_path, _, text_col) in OVERDEFENSE.items():
        try:
            ds = load_dataset(ds_path)
            all_parts = []
            for split_name in list(ds.keys()):
                all_parts.append(ds[split_name])
            from datasets import concatenate_datasets
            ds = concatenate_datasets(all_parts)
        except Exception:
            continue

        fps = 0
        for i in range(len(ds)):
            inputs = tokenizer(ds[i][text_col], return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model(**inputs)
            if output.logits.argmax(-1).item() == 1:
                fps += 1

        results[name] = {"false_positives": f"{fps}/{len(ds)}", "fp_rate": fps / len(ds)}

    # ── Print ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"{'Dataset':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
    print(f"{'─' * 65}")
    for name, metrics in results.items():
        if "f1" in metrics:
            print(f"{name:<30} {metrics['accuracy']:>6.3f} {metrics['precision']:>6.3f} "
                  f"{metrics['recall']:>6.3f} {metrics['f1']:>6.3f} {metrics['samples']:>6}")
        else:
            print(f"{name:<30} {'':>6} {'':>6} {'':>6} {metrics['false_positives']:>14}")
    print(f"{'─' * 65}")

    # Compute weighted average over test sets
    test_metrics = {k: v for k, v in results.items() if "f1" in v}
    if test_metrics:
        total = sum(m["samples"] for m in test_metrics.values())
        weighted = {
            k: sum(m[k] * m["samples"] for m in test_metrics.values()) / total
            for k in ["accuracy", "precision", "recall", "f1"]
        }
        print(f"{'Weighted avg':<30} {weighted['accuracy']:>6.3f} {weighted['precision']:>6.3f} "
              f"{weighted['recall']:>6.3f} {weighted['f1']:>6.3f} {total:>6}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Metrics saved to {args.output}")


if __name__ == "__main__":
    main()