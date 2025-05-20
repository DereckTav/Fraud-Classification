import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import yaml
import argparse
from transformers import BertTokenizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dataset import get_dataloader
from BERT_model.Sentiment_Analysis.models.model import AmazonReviewSentimentClassifier
from Datasets.processed_datasets.amazon_review_full_csv.normalize.ReviewData import reviewData

def get_predictions(model, data_loader, device):
    """
    Get predictions from model for all samples in data_loader
    """
    model.eval()
    all_titles, all_reviews, all_ratings = [], [], []
    all_preds, all_probs = [], []

    with torch.no_grad():
        for batch in data_loader:
            titles = batch.get('title', [])
            reviews = batch.get('review', [])
            ratings = batch.get('rating', [])

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )   

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            if titles:
                all_titles.extend(titles)
            if reviews:
                all_reviews.extend(reviews)
            if ratings:
                all_ratings.extend(ratings)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_titles, all_reviews, all_ratings, np.array(all_preds), np.array(all_probs)

def main(config_path, model_path, type, df_full, n):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract config parameters
    MODEL_NAME = config['model']['name']
    MAX_LEN = config['model']['max_len']
    N_CLASSES = config['model']['n_classes']
    BATCH_SIZE = config['training']['batch_size']
    N_WORKERS = config['training'].get('num_workers', 0)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"loading dataset... ")
    df_full = df_full.groupby('rating').sample(n=n, random_state=config['data']['random_state'])

    #shuffle
    df_full = df_full.sample(frac=1, random_state=config['data']['random_state']).reset_index(drop=True)

    print(f"Full dataset size: {len(df_full)}")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Get DataLoader for the entire dataset with shuffle=False
    full_loader = get_dataloader(
        df=df_full,
        tokenizer=tokenizer,
        max_length=MAX_LEN,
        batch_size=BATCH_SIZE,
        include_text=True,
        num_workers=N_WORKERS,
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

    # Run prediction on the entire dataset
    print("Running predictions on the entire dataset...")
    titles, reviews, ratings, preds, _ = get_predictions(model, full_loader, device)

    # Save predictions to CSV for the entire dataset
    predictions_df_full = pd.DataFrame({
        'polarity': preds,
        'rating': ratings,
        'title': titles if titles else [np.nan] * len(preds),
        'review': reviews if reviews else [np.nan] * len(preds),
    })

    predictions_path_full = f"Datasets/processed_datasets/amazon_review_full_csv/data/{type}.csv"
    predictions_df_full.to_csv(predictions_path_full, index=False)
    print(f"Predictions for the entire dataset saved to {predictions_path_full}")

    print("Classification of the entire dataset complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions with sentiment analysis model on the entire dataset')
    parser.add_argument('--config', type=str, default=os.path.join('BERT_model/Sentiment_Analysis/', 'config/config.yaml'), help='Path to config file')
    parser.add_argument('--model', type=str, default=os.path.join('BERT_model/Sentiment_Analysis/', 'model/title.bin'), help='Path to model weights')
    args = parser.parse_args()

    df_full = reviewData().get_data()
    main(args.config, args.model,'train', df_full, 20000)

    df_full = reviewData(type='test').get_data()
    main(args.config, args.model,'test', df_full, 25000)