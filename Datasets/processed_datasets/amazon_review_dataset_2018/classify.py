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
from Datasets.processed_datasets.amazon_review_dataset_2018.normalize.classification_data import classificationData

def get_predictions(model, data_loader, device):
    """
    Get predictions from model for all samples in data_loader
    """
    model.eval()
    all_reviews, all_ratings, all_product_types = [], [], []
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in data_loader:
            reviews = batch.get('review', [])
            ratings = batch.get('rating', [])
            product_types = batch.get('product_type', [])

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

            if reviews:
                all_reviews.extend(reviews)
            if ratings:
                all_ratings.extend(ratings)
            if product_types:
                all_product_types.extend(product_types)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_reviews, all_ratings, all_product_types, np.array(all_preds), np.array(all_probs)

def main(config_path, model_path_no_title, model_path_title, type, df_full, confidence_threshold=0.9):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Extract config parameters
    MODEL_NAME = config['model']['name']
    MAX_LEN = config['model']['max_len']
    N_CLASSES = config['model']['n_classes']
    BATCH_SIZE = config['training']['batch_size']
    N_WORKERS = config['training'].get('num_workers', 0)
    RANDOM_STATE = config['data']['random_state']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"loading dataset... ")

    # shuffle
    df_full = df_full.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Full dataset size: {len(df_full)}")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Get DataLoader for the entire dataset with include_text=True
    full_loader = get_dataloader(
        df=df_full,
        tokenizer=tokenizer,
        max_length=MAX_LEN,
        batch_size=BATCH_SIZE,
        include_text=True,
        num_workers=N_WORKERS,
    )

    # Load Model A (no title) - Assuming it was trained on reviews only
    model_no_title = AmazonReviewSentimentClassifier(N_CLASSES, MODEL_NAME).to(device)
    try:
        model_no_title.load_state_dict(torch.load(model_path_no_title, map_location=device))
        print(f"Model A (no title) loaded from {model_path_no_title}")
    except Exception as e:
        print(f"Failed to load model weights for Model A: {e}")
        return

    # Load Model B (title) - Assuming it was trained on reviews (and potentially title if your 'classifyDataset' was adapted)
    model_title = AmazonReviewSentimentClassifier(N_CLASSES, MODEL_NAME).to(device)
    try:
        model_title.load_state_dict(torch.load(model_path_title, map_location=device))
        print(f"Model B (title) loaded from {model_path_title}")
    except Exception as e:
        print(f"Failed to load model weights for Model B: {e}")
        return

    # Run prediction on the entire dataset with Model A (no title)
    print("Running predictions with Model A (no title) on the entire dataset...")
    reviews_a, ratings_a, product_types_a, preds_a, probs_a = get_predictions(model_no_title, full_loader, device)
    confidence_a = np.max(probs_a, axis=1)
    low_confidence_indices = np.where(confidence_a < confidence_threshold)[0]

    final_preds = np.copy(preds_a)
    final_probs = np.copy(probs_a)
    used_model_b = np.zeros(len(preds_a), dtype=bool)

    # Create a DataFrame for low confidence samples to get titles if available
    df_low_confidence = df_full.iloc[low_confidence_indices].reset_index(drop=True)

    if len(low_confidence_indices) > 0:
        # Create DataLoader for low confidence samples (including all text info)
        low_confidence_loader_b = get_dataloader(
            df=df_low_confidence,
            tokenizer=tokenizer,
            max_length=MAX_LEN,
            batch_size=BATCH_SIZE,
            include_text=True,
            num_workers=N_WORKERS,
        )

        # Run prediction with Model B (title) on low confidence samples
        print("Running predictions with Model B (title) on low confidence samples...")
        reviews_b, ratings_b, product_types_b, preds_b, probs_b = get_predictions(model_title, low_confidence_loader_b, device)

        # Update final predictions with Model B's predictions
        if len(preds_b) == len(low_confidence_indices):
            final_preds[low_confidence_indices] = preds_b
            final_probs[low_confidence_indices] = probs_b
            used_model_b[low_confidence_indices] = True
        else:
            print("Warning: Number of low confidence samples does not match predictions from Model B.")

    # Create the final predictions DataFrame
    predictions_df_final = pd.DataFrame({
        'review': reviews_a,
        'rating': ratings_a,
        'product_type': product_types_a,
        'predicted_type_model_a': preds_a,
        'confidence_model_a': confidence_a,
        'predicted_type_final': final_preds,
        'confidence_final': np.max(final_probs, axis=1),
        'used_model_b': used_model_b
    })

    predictions_path_final = f"ensemble_predictions_{type}.csv"
    predictions_df_final.to_csv(predictions_path_final, index=False)
    print(f"Final predictions saved to {predictions_path_final}")

    print("Classification complete using the ensemble approach.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions with ensemble of classification models')
    parser.add_argument('--config', type=str, default=os.path.join('BERT_model/Sentiment_Analysis/', 'config/config.yaml'), help='Path to config file')
    parser.add_argument('--model_no_title', type=str, default=os.path.join('BERT_model/Sentiment_Analysis/', 'model/no_title.bin'), help='Path to model weights trained without title (reviews only)')
    parser.add_argument('--model_title', type=str, default=os.path.join('BERT_model/Sentiment_Analysis/', 'model/title.bin'), help='Path to model weights trained with title (reviews + title)')
    args = parser.parse_args()

    # Assuming your classificationData returns a DataFrame with 'review', 'rating', and 'product_type' columns
    # and that 'get_dataloader' in your 'data.dataset' can handle this structure.
    df_full_train = classificationData().get_data()# Replace with your actual training data loading
    main(args.config, args.model_no_title, args.model_title, 'train', df_full_train, confidence_threshold=0.9)

    df_full_test = classificationData(type='test').get_data() # Replace with your actual test data loading
    main(args.config, args.model_no_title, args.model_title, 'test', df_full_test,confidence_threshold=0.9)