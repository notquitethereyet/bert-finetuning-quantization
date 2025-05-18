import argparse
import os
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import torch
import yaml
from datasets import load_dataset

def benchmark(model_dir, batch_size=32, n_batches=10, device=None, quantized=False):
    print(f"\nBenchmarking: {model_dir} (quantized={quantized})")
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load model
    if quantized:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, quantization_config=bnb_config, device_map="auto")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    # Load a batch of test data (URLs)
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    dataset = load_dataset("shawhin/phishing-site-classification", split="test")
    texts = [x['text'] for x in dataset.select(range(batch_size * n_batches))]
    # Tokenize all at once
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=config.get("max_length", 128), return_tensors="pt")
    # Run batches
    times = []
    from transformers import DistilBertForSequenceClassification
    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            input_batch = {k: v[start:end].to(device) for k, v in encodings.items()}
            # Remove token_type_ids for DistilBERT-based models
            if isinstance(model, DistilBertForSequenceClassification):
                input_batch.pop("token_type_ids", None)
            torch.cuda.synchronize() if device == 'cuda' else None
            t0 = time.time()
            _ = model(**input_batch)
            torch.cuda.synchronize() if device == 'cuda' else None
            t1 = time.time()
            times.append(t1 - t0)
    avg_time = sum(times) / len(times)
    print(f"Average inference time per batch (batch_size={batch_size}): {avg_time:.4f} s")
    print(f"Average time per sample: {avg_time / batch_size:.6f} s")
    return avg_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_batches', type=int, default=10)
    parser.add_argument('--device', type=str, default=None, help='cpu or cuda')
    args = parser.parse_args()
    # Benchmark all three models
    benchmark("phishing-bert-model", batch_size=args.batch_size, n_batches=args.n_batches, device=args.device)
    benchmark("distilled-student", batch_size=args.batch_size, n_batches=args.n_batches, device=args.device)
    benchmark("quantized-student", batch_size=args.batch_size, n_batches=args.n_batches, device=args.device, quantized=True)

if __name__ == "__main__":
    main()
