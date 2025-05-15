# BERT Phishing Site Classifier

## Overview
This project implements a phishing-site classifier using BERT (Bidirectional Encoder Representations from Transformers). The goal is to accurately distinguish between "Safe" and "Not Safe" URLs using state-of-the-art NLP techniques, with a reproducible, configurable, and production-ready pipeline.

---

## Model Architecture
- **Base Model:** [BERT-base-uncased](https://huggingface.co/bert-base-uncased)
    - 12-layer transformer encoder
    - 110M parameters
    - Pretrained on English Wikipedia and BooksCorpus
- **Classification Head:**
    - Linear layer mapping BERT's pooled output to 2 classes: `Safe` and `Not Safe`
- **Parameter Freezing:**
    - All BERT base model parameters are frozen except for the pooler layer and the classifier head. Only these are finetuned during training for efficiency and to prevent catastrophic forgetting.

---

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
