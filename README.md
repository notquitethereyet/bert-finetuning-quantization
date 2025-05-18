# BERT Phishing Site Classifier

## Overview
A phishing-site classifier using BERT and its compressed variants. Distinguishes "Safe" vs. "Not Safe" URLs with a reproducible, configurable, and production-ready pipeline.

---

## Model Variants
- **BERT (Teacher):** Highest accuracy, largest size.
- **Distilled Student:** Smaller, faster DistilBERT trained via knowledge distillation.
- **Quantized Student:** Distilled model compressed to 4-bit weights (HuggingFace BitsAndBytes).

| Model              | Accuracy | F1 Score | Disk Size | Speedup vs. BERT | Notes                       |
|--------------------|----------|----------|-----------|------------------|-----------------------------|
| BERT (Teacher)     | 0.887    | 0.888    | 440M      | 1.0x             | Best accuracy, slowest      |
| Distilled Student  | 0.904    | 0.904    | 203M      | 3.6x             | 2x smaller, fast            |
| Quantized Student  | 0.904    | 0.904    | 61M       | 3.3x             | Smallest, minimal loss      |

> **Note:** Disk size = total model directory (all files for inference).

---

## Dataset and Pipeline
- **Dataset:** [shawhin/phishing-site-classification](https://huggingface.co/datasets/shawhin/phishing-site-classification)
- **Splits:** `train`, `validation`, `test`
- **Features:** `text` (URL), `labels` (0: Safe, 1: Not Safe)
- **Pipeline:**
  1. **Train Teacher:** Finetune BERT with HuggingFace Trainer (early stopping, best checkpoint).
  2. **Distill Student:** `distill.py` trains DistilBERT from teacher logits + ground truth (custom loss). 
  3. **Quantize:** `quantize_student.py` compresses the distilled model to 4-bit.
  4. **Inference:** All models support batch/single-URL inference. Results: accuracy, F1, confusion matrix, classification report.

> **Note:** DistilBERT and quantized models do **not** use `token_type_ids`. Scripts handle this automatically.
          if "pooler" in name:
              param.requires_grad = True
      ```
4. **Training/Distillation:**
    - Teacher: HuggingFace's `Trainer` API with early stopping and best model saving (metric configurable via `config.yaml`).
    - Student: Custom PyTorch loop in `distill.py` using both hard and soft targets, early stopping, and best model restoration. Best model logic now robust to metric direction.
5. **Quantization:**
    - Loads the distilled student model in 4-bit mode using `BitsAndBytesConfig` via `quantize_student.py`. CLI overrides for model/output dir now supported.
6. **Saving:**
    - All models and tokenizers are saved to their respective output directories.
7. **Configuration & Overrides:**
    - All scripts read from `config.yaml` by default, but `inference.py` and `quantize_student.py` support command-line overrides for key arguments (e.g., `--model_dir`, `--output_dir`).
    - Distillation and quantization scripts expect relevant keys (e.g., `teacher_model`, `student_model`, `distill_output_dir`, etc.) in `config.yaml`.
    - Error handling improved for missing config keys.

---

## Inference
- All models support both batch and single-URL inference.
- Use `--model_dir` to specify which model to load (teacher, distilled, or quantized). This overrides the value in `config.yaml` if provided.
- Outputs accuracy, F1, confusion matrix, and classification report for batch inference.
- `--split` and `--input_text` can also be passed as CLI arguments if supported by the script, or set in `config.yaml`.
- `--config` can be omitted if your script always uses `config.yaml` in the current directory.

**Example Inference Commands:**
```bash
# Teacher (full BERT)
uv run inference.py --model_dir phishing-bert-model

# Distilled Student
uv run inference.py --model_dir distilled-student

# Quantized Student
uv run inference.py --model_dir quantized-student

# Single URL inference
uv run inference.py --input_text "http://example.com" --model_dir quantized-student
```

> **Note:** The script uses the test set by default. The `--split` argument is not required.

---

## Quantization
- To quantize a distilled student model safely:
    ```bash
    python quantize_student.py --model_dir distilled-student --output_dir quantized-student
    ```
  - If you do **not** specify `--output_dir`, the quantized model will always be saved to `quantized-student` (never overwriting your parent model).
  - If you try to set `output_dir` to the same location as your input model (e.g., `phishing-bert-model` or `distilled-student`), the script will print an **error and exit**—your parent model is always protected.
  - This prevents accidental overwrites of your original models.
  - Uses 4-bit NF4 quantization via HuggingFace `BitsAndBytesConfig`.

> **Warning:**
> - If you see an error about `output_dir` matching `model_dir`, the script is blocking you from overwriting your parent model. To resolve, specify a unique `--output_dir` (or set it in `config.yaml`).
> - Always use a unique folder for each quantized model. **Never** use your original model's directory as the output.

> **Troubleshooting:**
> - If you see `[ERROR] output_dir (...) is the same as model_dir (...)` and the script exits, you must choose a different output directory.
> - This protection is strict: you cannot bypass it unless you modify the script.

- Required config keys for distillation and quantization:
    - `distill_output_dir`: Where to save the distilled student model
    - `model_dir`: Path to model for quantization (can be set to distilled student)
    - `output_dir`: Where to save the quantized model
    - Other training hyperparameters as needed (see sample config)
- CLI arguments always override config file values.
- Example snippet:
    ```yaml
    teacher_model: phishing-bert-model
    student_model: distilbert-base-uncased
    distill_output_dir: distilled-student
    model_dir: distilled-student
    output_dir: quantized-student
    max_length: 128
    train_batch_size: 8
    eval_batch_size: 8
    num_epochs: 50
    learning_rate: 0.0002
    metric_for_best_model: f1
    early_stopping_patience: 5
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
No license lil bro.

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
python inference.py --config config.yaml --split test
python inference.py --config config.yaml --input_text "http://example.com"
```

---

## Requirements
- Python 3.8+
- transformers
- datasets
- scikit-learn
- numpy
- pyyaml
- torch (with CUDA for GPU)
- bitsandbytes (for quantization)

**Install (recommended: [uv](https://github.com/astral-sh/uv))**
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
uv init
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Evaluation Metrics
- **Accuracy**: Proportion of correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Breakdown of true/false positives/negatives
- **Classification Report**: Precision, recall, F1, support per class

---

## Extensibility
- Change `model_name` in `config.yaml` for other transformers.
- Adjust batch size, learning rate, epochs in config.
- Add metrics/callbacks as needed.
- Suitable for other binary sequence classification tasks.

---

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
