import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, AutoTokenizer
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Quantize a student model using 4-bit quantization.")
    parser.add_argument('--model_dir', type=str, required=True, help='Path to distilled student model directory')
    parser.add_argument('--output_dir', type=str, default='quantized-student', help='Where to save the quantized model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        device_map="auto",
        quantization_config=nf4_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Quantized model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
