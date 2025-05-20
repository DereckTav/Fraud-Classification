import argparse
import torch
import yaml
import os
import sys
from transformers import AutoModelForSequenceClassification, BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path):
    """Load configuration from YAML file with error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config

def load_model(model_name):
    """Load pre-trained model with error handling."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return model

def predict(model, tokenizer, rating, review, max_len):
    """Make prediction using the model."""
    encoded = tokenizer(
        text=str(rating),
        text_pair=str(review),
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)
    token_type_ids = encoded.get('token_type_ids', None)
    
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        
    return pred, probs.cpu().numpy().flatten()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fraud Detection CLI")
    parser.add_argument("--rating", type=str, help="Rating (e.g., '5')")
    parser.add_argument("--review", type=str, help="Review text")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (ignored)")
    args = parser.parse_args()

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config) if not os.path.isabs(args.config) else args.config
        config = load_config(config_path)

        tokenizer_name = config["model_name"]
        model_name = config["hugging_face_repo"]["model_name"]
        max_len = config.get("max_len", 512)
        class_names = {int(k): v for k, v in config["class_names"].items()}

        rating = args.rating if args.rating else input("Enter rating (e.g. 5): ").strip()
        review = args.review if args.review else input("Enter review text: ").strip()

        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        model = load_model(model_name)
        pred, probs = predict(model, tokenizer, rating, review, max_len)
        label = class_names.get(pred, f"Class {pred}")

        print(f"\nPrediction: {label} (Confidence: {probs[pred]:.4f})")
        print(f"All probabilities: {dict(zip(class_names.values(), probs))}")
        print(f'this review is {label}')

    except FileNotFoundError as e:
        print(f"\nERROR: {str(e)}")
        return 1
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
