import logging
import argparse
import yaml
from typing import Any, Dict
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import transformers
import os

def get_logger() -> logging.Logger:
    """Set up and return a logger for the script."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config: Dict[str, Any]) -> None:
    """Validate config values."""
    assert config["max_length"] > 0, "max_length must be positive"
    assert config["train_batch_size"] > 0, "train_batch_size must be positive"
    assert config["eval_batch_size"] > 0, "eval_batch_size must be positive"
    assert config["num_epochs"] > 0, "num_epochs must be positive"
    assert os.path.isdir(config["logging_dir"]) or config["logging_dir"] == './logs', "logging_dir must be a valid directory path or './logs'"

def main() -> None:
    """
    Main training routine for BERT phishing-site classification.
    Loads configuration, prepares data, trains and evaluates the model.
    """
    config = load_config("config.yaml")
    validate_config(config)

    logger = get_logger()
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")

    # Load dataset
    dataset = load_dataset("shawhin/phishing-site-classification")

    # Model and tokenizer
    model_path = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    id2label = {0: "Safe", 1: "Not Safe"}
    label2id = {"Safe": 0, "Not Safe": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    # (Optional) Freeze all base model parameters except pooler
    for name, param in model.base_model.named_parameters():
        param.requires_grad = False
        if "pooler" in name:
            param.requires_grad = True

    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize the input examples using the loaded tokenizer."""
        if "text" not in examples:
            raise KeyError("'text' field not found in input examples.")
        return tokenizer(examples["text"], truncation=True, max_length=config["max_length"])

    tokenized_data = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        """Compute accuracy and F1 metrics for evaluation."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='binary')
        return {"accuracy": acc, "f1": f1}

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["num_epochs"],
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config["metric_for_best_model"],
        save_total_limit=config["save_total_limit"],
        report_to=config["report_to"],
        logging_dir=config["logging_dir"],
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])],
    )

    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    predictions = trainer.predict(tokenized_data["test"])
    logits = predictions.predictions
    labels = predictions.label_ids
    metrics = compute_metrics((logits, labels))
    logger.info(f"Test metrics: {metrics}")

    # Resource cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Model and tokenizer
    model_path = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    id2label = {0: "Safe", 1: "Not Safe"}
    label2id = {"Safe": 0, "Not Safe": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    # (Optional) Freeze all base model parameters except pooler
    for name, param in model.base_model.named_parameters():
        param.requires_grad = False
        if "pooler" in name:
            param.requires_grad = True

    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize the input examples using the loaded tokenizer."""
        if "text" not in examples:
            raise KeyError("'text' field not found in input examples.")
        return tokenizer(examples["text"], truncation=True, max_length=config["max_length"])

    tokenized_data = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        """Compute accuracy and F1 metrics for evaluation."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='binary')
        return {"accuracy": acc, "f1": f1}

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["num_epochs"],
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config["metric_for_best_model"],
        save_total_limit=config["save_total_limit"],
        report_to=config["report_to"],
        logging_dir=config["logging_dir"],
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])],
    )

    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    predictions = trainer.predict(tokenized_data["test"])
    logits = predictions.predictions
    labels = predictions.label_ids
    metrics = compute_metrics((logits, labels))
    logger.info(f"Test metrics: {metrics}")

    # Resource cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()