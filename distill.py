import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification, DistilBertConfig
import argparse
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_function(examples, tokenizer, max_length):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=max_length)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    return accuracy, precision, recall, f1

def distillation_loss(student_logits, teacher_logits, true_labels, temperature, alpha):
    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
    student_soft = nn.functional.log_softmax(student_logits / temperature, dim=1)
    distill_loss = nn.functional.kl_div(student_soft, soft_targets, reduction='batchmean') * (temperature ** 2)
    hard_loss = nn.CrossEntropyLoss()(student_logits, true_labels)
    return alpha * distill_loss + (1.0 - alpha) * hard_loss

def main():
    # All parameters must be set in config.yaml
    config = load_config("config.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Required keys in config.yaml:
    # - teacher_model: Path or HF hub id for teacher model
    # - student_model: HF model id for student
    # - output_dir: Where to save the student model
    # - n_layers: Number of layers for student
    # - n_heads: Number of attention heads for student
    # - max_length, train_batch_size, eval_batch_size, learning_rate, num_epochs, etc.

    teacher_model_path = config["teacher_model"]
    student_model_id = config.get("student_model", "distilbert-base-uncased")
    output_dir = config.get("output_dir", "distilled-student")
    n_layers = config.get("n_layers", 4)
    n_heads = config.get("n_heads", 8)

    # Load dataset
    data = load_dataset("shawhin/phishing-site-classification")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    max_length = config.get("max_length", 128)
    def preprocess(examples):
        return preprocess_function(examples, tokenizer, max_length)
    tokenized_data = data.map(preprocess, batched=True)
    tokenized_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # DataLoaders
    train_loader = DataLoader(tokenized_data['train'], batch_size=config.get('train_batch_size', 8), shuffle=True)
    val_loader = DataLoader(tokenized_data['validation'], batch_size=config.get('eval_batch_size', 8))
    test_loader = DataLoader(tokenized_data['test'], batch_size=config.get('eval_batch_size', 8))

    # Teacher
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_path).to(device)
    teacher_model.eval()

    # Student
    student_config = DistilBertConfig.from_pretrained(student_model_id, n_layers=n_layers, n_heads=n_heads, num_labels=2)
    student_model = DistilBertForSequenceClassification.from_pretrained(student_model_id, config=student_config).to(device)

    # Training setup
    optimizer = optim.Adam(student_model.parameters(), lr=config.get('learning_rate', 2e-4))
    num_epochs = config.get('num_epochs', 5)
    temperature = 2.0
    alpha = 0.5

    metric_for_best_model = config.get('metric_for_best_model', 'eval_loss')
    patience = config.get('early_stopping_patience', 5)
    # Initialize best_metric for correct direction
    if metric_for_best_model in ['accuracy', 'f1']:
        best_metric = float('-inf')  # maximize
        compare = lambda current, best: current > best
    else:
        best_metric = float('inf')   # minimize
        compare = lambda current, best: current < best
    best_epoch = 0
    best_model_state = None
    epochs_no_improve = 0

    def get_val_metric(acc, prec, rec, f1):
        # Support 'eval_loss', 'accuracy', 'f1', etc. Default: maximize accuracy/f1, minimize loss.
        if metric_for_best_model == 'accuracy':
            return acc
        elif metric_for_best_model == 'f1':
            return f1
        elif metric_for_best_model == 'eval_loss':
            # We don't compute loss here, so use -f1 as a proxy for minimizing loss
            return -f1
        else:
            return f1

    for epoch in range(num_epochs):
        student_model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask=attention_mask).logits
            student_logits = student_model(input_ids, attention_mask=attention_mask).logits
            loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Evaluate
        acc, prec, rec, f1 = evaluate_model(student_model, val_loader, device)
        val_metric = get_val_metric(acc, prec, rec, f1)
        print(f"Epoch {epoch+1}: Val Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        if compare(val_metric, best_metric):
            best_metric = val_metric
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in student_model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch+1} (best_metric={best_metric:.4f})")
            break
    # Restore best model
    if best_model_state is not None:
        student_model.load_state_dict(best_model_state)
    # Final evaluation
    acc, prec, rec, f1 = evaluate_model(student_model, test_loader, device)
    print(f"Test: Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    # Save student model
    distill_output_dir = config.get('distill_output_dir', 'distilled-student')
    os.makedirs(distill_output_dir, exist_ok=True)
    student_model.save_pretrained(distill_output_dir)
    tokenizer.save_pretrained(distill_output_dir)

if __name__ == "__main__":
    main()
