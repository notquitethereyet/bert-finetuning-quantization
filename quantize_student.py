import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, AutoTokenizer
import os

def main():
    """
    Quantize a student model using 4-bit NF4 quantization. Supports optional CLI overrides for model_dir and output_dir.
    """
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Quantize a student model with 4-bit NF4 quantization")
    parser.add_argument('--model_dir', type=str, default=None, help='Path to distilled student model directory (overrides config.yaml)')
    parser.add_argument('--output_dir', type=str, default=None, help='Where to save the quantized model (overrides config.yaml)')
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # CLI overrides config.yaml if provided
    model_dir = args.model_dir if args.model_dir is not None else config.get("model_dir", "distilled-student")
    # Robust output_dir logic: only allow explicit, safe output_dir
    if args.output_dir is not None:
        if os.path.abspath(args.output_dir) == os.path.abspath(model_dir):
            print(f"[ERROR] output_dir ({args.output_dir}) is the same as model_dir ({model_dir}). Refusing to overwrite parent model. Please specify a different --output_dir.")
            exit(1)
        output_dir = args.output_dir
    else:
        # Only use config.yaml output_dir if it is set, not equal to model_dir, and not a parent/child of model_dir
        config_output_dir = config.get("output_dir", None)
        if (
            config_output_dir is None or
            os.path.abspath(config_output_dir) == os.path.abspath(model_dir) or
            os.path.abspath(model_dir).startswith(os.path.abspath(config_output_dir) + os.sep) or
            os.path.abspath(config_output_dir).startswith(os.path.abspath(model_dir) + os.sep)
        ):
            print(f"[WARNING] output_dir in config.yaml is missing, matches, or is a parent/child of model_dir ({model_dir}). Saving quantized model to 'quantized-student' instead.")
            output_dir = "quantized-student"
        else:
            output_dir = config_output_dir
    # Final safety check -- do NOT proceed if output_dir matches model_dir
    if os.path.abspath(output_dir) == os.path.abspath(model_dir):
        print(f"[ERROR] output_dir ({output_dir}) is the same as model_dir ({model_dir}). Refusing to overwrite parent model. Please specify a different output_dir.")
        exit(1)
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        device_map="auto",
        quantization_config=nf4_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized model saved to {output_dir}")

if __name__ == "__main__":
    main()
