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
        # Remove token_type_ids if model does not accept it (e.g., DistilBERT)
        try:
            with torch.no_grad():
                outputs = model(**inputs)
        except TypeError as e:
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
                with torch.no_grad():
                    outputs = model(**inputs)
            else:
                raise e
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
    import argparse
    parser = argparse.ArgumentParser(description="Phishing-site BERT inference")
    parser.add_argument('--model_dir', type=str, default=None, help='Directory of model to use for inference (overrides config.yaml)')
    args = parser.parse_args()

    config = load_config("config.yaml")
    validate_config(config)

    split = config.get("split", "test")
    input_text = config.get("input_text", None)
    output_file = config.get("output_file", None)
    # Use CLI argument if provided, else config.yaml
    model_dir = args.model_dir if args.model_dir is not None else config.get("model_dir", config["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    id2label = get_id2label(model)
    max_length = config["max_length"]
    batch_size = config["eval_batch_size"]

    if input_text:
        # Single example inference
        texts = [input_text]
        preds, probs = predict_batch(texts, tokenizer, model, device, batch_size=1, max_length=max_length)
        print(f"Input: {input_text}")
        print(f"Prediction: {id2label[preds[0]]} (prob={probs[0][preds[0]]:.3f})")
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({"input": input_text, "prediction": id2label[preds[0]], "prob": float(probs[0][preds[0]])}, f)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    # Dataset inference
    dataset = load_dataset("shawhin/phishing-site-classification")
    data = dataset[split]
    urls = data["text"]
    true_labels = data["labels"]
    preds, probs = predict_batch(urls, tokenizer, model, device, batch_size=batch_size, max_length=max_length)

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='binary')
    cm = confusion_matrix(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=[id2label[0], id2label[1]])

    print(f"Accuracy on {split} set: {acc:.3f}")
    print(f"F1 score on {split} set: {f1:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    if output_file:
        results = {
            "accuracy": acc,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "predictions": [int(p) for p in preds]
        }
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Resource cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()