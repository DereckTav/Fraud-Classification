import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import yaml
import argparse
from transformers import BertTokenizer


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import project modules
from data.dataset_no_title import get_dataloader
from models.model import AmazonReviewSentimentClassifier
from utils.metrics import calculate_metrics, plot_confusion_matrix
from Datasets.processed_datasets.amazon_review_sentiment.sentiment_data import SentimentData  

def get_predictions(model, data_loader, device):
    """
    Get predictions from model for all samples in data_loader
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device) 
            
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def main(config_path, model_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories if they don't exist
    os.makedirs('BERT_model/Sentiment_Analysis/outputs/predictions', exist_ok=True)
    os.makedirs('BERT_model/Sentiment_Analysis/outputs/figures', exist_ok=True)
    
    # Extract config parameters
    MODEL_NAME = config['model']['name']
    MAX_LEN = config['model']['max_len']
    N_CLASSES = config['model']['n_classes']
    BATCH_SIZE = config['training']['batch_size']
    N_WORKERS = config['training'].get('num_workers', 0)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test dataset...")
    sentiment_data = SentimentData(type='test')
    df_test = sentiment_data.get_data()
    df_test = df_test.groupby('polarity').sample(n=15000, random_state=config['data']['random_state'])
    df_test = df_test.sample(frac=1, random_state=config['data']['random_state']).reset_index(drop=True)
    df_test['polarity'] = df_test['polarity'].astype(int)
    
    print(f"Test set size: {len(df_test)}")
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Get DataLoader
    test_loader = get_dataloader(
        df=df_test,
        tokenizer=tokenizer,
        max_length=MAX_LEN,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS
    )
    
    # Load model
    model = AmazonReviewSentimentClassifier(N_CLASSES, MODEL_NAME).to(device)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Failed to load model weights from {model_path}: {e}")
        return
    
    # Run prediction
    print("Running predictions...")
    preds, probs, labels = get_predictions(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(labels, preds)
    print("\nTest Set Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.capitalize()}: {metric_value:.4f}")
    
    # Plot confusion matrix
    class_names = ['Negative', 'Positive']  # Assuming binary classification
    cm_path = "BERT_model/Sentiment_Analysis/outputs/figures/confusion_no_title_matrix(title model).png"
    plot_confusion_matrix(labels, preds, class_names=class_names, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'true_label': labels,
        'predicted_label': preds,
        'confidence': np.max(probs, axis=1)
    })

    
    # Save predictions to CSV
    predictions_path = "BERT_model/Sentiment_Analysis/outputs/predictions/test_no_title_predictions(title model).csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Output sample predictions
    print("\nSample Predictions:")
    sample_size = min(5, len(predictions_df))
    for i in range(sample_size):
        print(f"\nSample {i+1}:")
        print(f"True Label: {predictions_df['true_label'].iloc[i]} ({class_names[predictions_df['true_label'].iloc[i]]})")
        print(f"Predicted Label: {predictions_df['predicted_label'].iloc[i]} ({class_names[predictions_df['predicted_label'].iloc[i]]})")
        print(f"Confidence: {predictions_df['confidence'].iloc[i]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions with sentiment analysis model')
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/config.yaml'), help='Path to config file')
    parser.add_argument('--model', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/saved/best_model_state_(loss).bin'), help='Path to model weights')
    args = parser.parse_args()
    
    main(args.config, args.model)