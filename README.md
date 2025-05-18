# BERT Phishing Site Classifier

## Overview
This project implements a phishing-site classifier using BERT (Bidirectional Encoder Representations from Transformers). The goal is to accurately distinguish between "Safe" and "Not Safe" URLs using state-of-the-art NLP techniques, with a reproducible, configurable, and production-ready pipeline.

---

## Model Variants and Compression Pipeline

This repository supports three model variants:
- **Full BERT (Teacher):** Highest accuracy, largest size.
- **Distilled Student:** Smaller DistilBERT-based model, trained via knowledge distillation from the teacher.
- **Quantized Student:** The distilled model further compressed to 4-bit weights using HuggingFace BitsAndBytes, minimizing memory and inference costs.

### Model Comparison (Test Set Results)
| Model              | Accuracy | F1 Score | Disk Size   | Notable Tradeoffs          |
|--------------------|----------|----------|-------------|---------------------------|
| BERT (Teacher)     | 0.887    | 0.888    | 1.3G        | Best accuracy, slowest    |
| Distilled Student  | 0.836    | 0.854    | 203M        | ~10x faster, 6x smaller   |
| Quantized Student  | 0.838    | 0.856    | 61M         | Minimal loss, smallest    |

> **Note:** Disk sizes are measured using `du -sh <model_dir>` and reflect the total space used by each model directory, including all necessary files for inference.

#### Confusion Matrices and Reports
- See below for full classification reports and confusion matrices for each model.

---

## Technical Pipeline

### 1. Training the Teacher (BERT)
- Standard supervised finetuning with HuggingFace Trainer.
- Early stopping and best model checkpointing using `config.yaml` parameters.

### 2. Knowledge Distillation
- **Script:** `distill.py`
- The student (DistilBERT) learns from both ground truth and the teacher's soft logits.
- Custom distillation loss: weighted sum of KL divergence (soft targets) and cross-entropy (hard targets).
- Early stopping and best model saving supported.

### 3. Quantization
- **Script:** `quantize_student.py`
- Uses HuggingFace `BitsAndBytesConfig` to load the distilled model in 4-bit (nf4) mode.
- Results in a 3-4x reduction in disk and memory usage with negligible accuracy loss.

---

## Dataset
- **Source:** [shawhin/phishing-site-classification](https://huggingface.co/datasets/shawhin/phishing-site-classification)
- **Splits:** `train`, `validation`, `test`
- **Features:**
    - `text`: URL string
    - `labels`: 0 (Safe), 1 (Not Safe)

---

## Finetuning & Distillation Process
1. **Configuration:**
    - All hyperparameters and paths are stored in `config.yaml` for easy reproducibility (see below).
2. **Tokenization:**
    - URLs are tokenized using the appropriate tokenizer, with truncation and padding to `max_length`.
3. **Parameter Freezing:**
    - All BERT layers except the pooler are frozen:
      ```python
      for name, param in model.base_model.named_parameters():
          param.requires_grad = False
          if "pooler" in name:
              param.requires_grad = True
      ```
4. **Training/Distillation:**
    - Teacher: HuggingFace's `Trainer` API with early stopping and best model saving.
    - Student: Custom PyTorch loop in `distill.py` using both hard and soft targets, early stopping, and best model restoration.
5. **Quantization:**
    - Loads the distilled student model in 4-bit mode using `BitsAndBytesConfig`.
6. **Saving:**
    - All models and tokenizers are saved to their respective output directories.

---

## Inference
- All models support both batch and single-URL inference.
- Use `--model_dir` to specify which model to load (teacher, distilled, or quantized).
- Outputs accuracy, F1, confusion matrix, and classification report for batch inference.

**Example Inference Commands:**
```bash
# Teacher (full BERT)
python inference.py --config config.yaml --split test --model_dir phishing-bert-model/

# Distilled Student
python inference.py --config config.yaml --split test --model_dir distilled-student

# Quantized Student
python inference.py --config config.yaml --split test --model_dir quantized-student

# Single URL inference
python inference.py --config config.yaml --input_text "http://example.com" --model_dir quantized-student
```

---

## Example Results

### Teacher (BERT)
```
Accuracy on test set: 0.887
F1 score on test set: 0.888
Confusion Matrix:
[[197  24]
 [ 27 202]]

Classification Report:
              precision    recall  f1-score   support

        Safe       0.88      0.89      0.89       221
    Not Safe       0.89      0.88      0.89       229

    accuracy                           0.89       450
   macro avg       0.89      0.89      0.89       450
weighted avg       0.89      0.89      0.89       450
```

### Distilled Student
```
Accuracy on test set: 0.836
F1 score on test set: 0.854
Confusion Matrix:
[[160  61]
 [ 13 216]]

Classification Report:
              precision    recall  f1-score   support

     LABEL_0       0.92      0.72      0.81       221
     LABEL_1       0.78      0.94      0.85       229

    accuracy                           0.84       450
   macro avg       0.85      0.83      0.83       450
weighted avg       0.85      0.84      0.83       450
```

### Quantized Student
```
Accuracy on test set: 0.838
F1 score on test set: 0.856
Confusion Matrix:
[[160  61]
 [ 12 217]]

Classification Report:
              precision    recall  f1-score   support

     LABEL_0       0.93      0.72      0.81       221
     LABEL_1       0.78      0.95      0.86       229

    accuracy                           0.84       450
   macro avg       0.86      0.84      0.84       450
weighted avg       0.85      0.84      0.84       450
```

---

## Configuration File (`config.yaml`)
All major parameters are stored in `config.yaml`:
```yaml
model_name: bert-base-uncased
output_dir: phishing-bert-model
max_length: 128
train_batch_size: 8
eval_batch_size: 8
num_epochs: 50
learning_rate: 0.0002
logging_dir: ./logs
save_total_limit: 2
metric_for_best_model: eval_loss
report_to: none
early_stopping_patience: 5
```

---

## Example Pipeline Usage

**Train Teacher:**
```bash
python train.py --config config.yaml
```
**Distill Student:**
```bash
python distill.py --teacher_model phishing-bert-model/ --output_dir distilled-student
```
**Quantize Student:**
```bash
python quantize_student.py --model_dir distilled-student --output_dir quantized-student
```
**Inference:** (see above)

---

## Extensibility
- Easily switch to other transformer models by changing `model_name` in `config.yaml`.
- Adjust batch sizes, learning rate, or number of epochs in the config.
- Add more metrics or callbacks in the training script as needed.
- Use for other binary sequence classification tasks with minimal changes.
- All scripts support config-driven workflow for reproducibility.

---

## Requirements
- Python 3.8+
- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- scikit-learn
- numpy
- pyyaml
- torch (with CUDA for GPU support)
- bitsandbytes (for quantization)

Install dependencies (recommended: [uv](https://github.com/astral-sh/uv)):

> **uv** is a super-fast, modern Python package manager and virtual environment tool. It is a drop-in replacement for pip/pip-tools/virtualenv, providing vastly improved speed and reliability.

**Quickstart:**
```bash
# Install uv (if not already installed)
curl -Ls https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment (optional but recommended)
uv init
uv venv
source .venv/bin/activate

# Install all dependencies from requirements.txt
uv pip install -r requirements.txt
```

---

## License
[Add your license here]

## Dataset
- **Source:** [shawhin/phishing-site-classification](https://huggingface.co/datasets/shawhin/phishing-site-classification)
- **Splits:** `train`, `validation`, `test`
- **Features:**
    - `text`: URL string
    - `labels`: 0 (Safe), 1 (Not Safe)

---

## Finetuning Process
1. **Configuration:**
    - All hyperparameters and paths are stored in `config.yaml` for easy reproducibility (see below).
2. **Tokenization:**
    - URLs are tokenized using BERT's tokenizer, with truncation and padding to `max_length`.
3. **Parameter Freezing:**
    - All BERT layers except the pooler are frozen:
      ```python
      for name, param in model.base_model.named_parameters():
          param.requires_grad = False
          if "pooler" in name:
              param.requires_grad = True
      ```
4. **Training:**
    - Performed using HuggingFace's `Trainer` API
    - Early stopping is used to prevent overfitting
    - Metrics: Accuracy and F1-score
5. **Saving:**
    - Both model and tokenizer are saved to the configured output directory

---

## Inference
- Supports both batch inference on dataset splits and single-URL inference.
- Loads the trained model and tokenizer from the output directory specified in `config.yaml`.
- Outputs accuracy, F1, confusion matrix, and classification report for batch inference.

---

## Configuration File (`config.yaml`)
All major parameters are stored in `config.yaml`:
```yaml
model_name: bert-base-uncased
output_dir: phishing-bert-model
max_length: 128
train_batch_size: 8
eval_batch_size: 8
num_epochs: 50
learning_rate: 0.0002
logging_dir: ./logs
save_total_limit: 2
metric_for_best_model: eval_loss
report_to: none
early_stopping_patience: 5
```

---

## Example Commands
**Training:**
```bash
python train.py --config config.yaml
```

**Inference on test split:**
```bash
python inference.py --config config.yaml --split test
```

**Inference on a custom URL:**
```bash
python inference.py --config config.yaml --input_text "http://example.com"
```

---

## Evaluation Metrics
- **Accuracy**: Proportion of correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Breakdown of true/false positives/negatives
- **Classification Report**: Precision, recall, F1, and support for each class

---

## Extensibility
- Easily switch to other transformer models by changing `model_name` in `config.yaml`.
- Adjust batch sizes, learning rate, or number of epochs in the config.
- Add more metrics or callbacks in the training script as needed.
- Use for other binary sequence classification tasks with minimal changes.

---

## Requirements
- Python 3.8+
- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- scikit-learn
- numpy
- pyyaml
- torch (with CUDA for GPU support)

Install dependencies (recommended: [uv](https://github.com/astral-sh/uv)):

> **uv** is a super-fast, modern Python package manager and virtual environment tool. It is a drop-in replacement for pip/pip-tools/virtualenv, providing vastly improved speed and reliability.

**Quickstart:**
```bash
# Install uv (if not already installed)
curl -Ls https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment (optional but recommended)
uv init
uv venv
source .venv/bin/activate

# Install all dependencies from requirements.txt
uv pip install -r requirements.txt
```

---

## License
[Add your license here]
