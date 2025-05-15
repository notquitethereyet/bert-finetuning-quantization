import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
import sys
import yaml
from typing import Any, Dict, List
import os

def get_id2label(model: AutoModelForSequenceClassification) -> Dict[int, str]:
    """Extract id2label mapping from model config or provide default."""
    if hasattr(model.config, 'id2label') and model.config.id2label:
        return {int(k): v for k, v in model.config.id2label.items()}
    return {0: "Safe", 1: "Not Safe"}

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config: Dict[str, Any]) -> None:
    """Validate config values."""
    assert config["max_length"] > 0, "max_length must be positive"
    assert config["eval_batch_size"] > 0, "eval_batch_size must be positive"
    assert os.path.isdir(config["output_dir"]) or config["output_dir"] == 'phishing-bert-model', "output_dir must be a valid directory path or 'phishing-bert-model'"

def predict_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128
) -> (List[int], List[List[float]]):
    """
    Batch predict labels and probabilities for a list of texts.
    Args:
        texts: List of input strings.
        tokenizer: Tokenizer instance.
        model: Model instance.
        device: torch.device.
        batch_size: Batch size for inference.
        max_length: Max tokenization length.
    Returns:
        Tuple of predictions and probabilities.
    """
    all_preds = []
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_probs.extend(probs.cpu().numpy())
    return all_preds, all_probs

def main() -> None:
    """
    Main inference routine for phishing-site BERT classifier.
    Loads configuration, runs inference on dataset or custom text, and prints/saves results.
    """
    parser = argparse.ArgumentParser(description="Inference for phishing BERT model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--split', type=str, default='test', choices=['train','validation','test'], help='Dataset split to use')
    parser.add_argument('--input_text', type=str, default=None, help='Custom text for inference (overrides dataset)')
    parser.add_argument('--output_file', type=str, default=None, help='File to save predictions/metrics')
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["output_dir"])
    model = AutoModelForSequenceClassification.from_pretrained(config["output_dir"])
    model.to(device)
    model.eval()
    id2label = get_id2label(model)
    max_length = config["max_length"]
    batch_size = config["eval_batch_size"]

    if args.input_text:
        # Single example inference
        texts = [args.input_text]
        preds, probs = predict_batch(texts, tokenizer, model, device, batch_size=1, max_length=max_length)
        print(f"Input: {args.input_text}")
        print(f"Prediction: {id2label[preds[0]]} (prob={probs[0][preds[0]]:.3f})")
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump({"input": args.input_text, "prediction": id2label[preds[0]], "prob": float(probs[0][preds[0]])}, f)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    # Dataset inference
    dataset = load_dataset("shawhin/phishing-site-classification")
    data = dataset[args.split]
    urls = data["text"]
    true_labels = data["labels"]
    preds, probs = predict_batch(urls, tokenizer, model, device, batch_size=batch_size, max_length=max_length)

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='binary')
    cm = confusion_matrix(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=[id2label[0], id2label[1]])

    print(f"Accuracy on {args.split} set: {acc:.3f}")
    print(f"F1 score on {args.split} set: {f1:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    if args.output_file:
        results = {
            "accuracy": acc,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "predictions": [int(p) for p in preds]
        }
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Resource cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()