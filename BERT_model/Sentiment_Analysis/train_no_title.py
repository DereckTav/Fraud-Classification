import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import defaultdict
import yaml
import os
import argparse
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import project modules
from data.dataset_no_title import get_dataloader
from models.model import AmazonReviewSentimentClassifier
from utils.metrics import plot_training_history
from Datasets.processed_datasets.amazon_review_sentiment.sentiment_data import SentimentData

def train_model(model, data_loader, loss_fn, optimizer, warm_scheduler, n_examples, device, use_amp=False):
    """
    Train the model for one epoch
    """
    model.train()
    losses = []
    correct_predictions = 0

    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = loss_fn(logits, labels)
            losses.append(loss.item())
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        
        optimizer.zero_grad()

        # Backward pass, with or without AMP
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if warm_scheduler:
            warm_scheduler.step()
            
    return np.mean(losses), correct_predictions / n_examples

def eval_model(model, data_loader, loss_fn, device, n_examples, use_amp=False):
    """
    Evaluate the model on validation data
    """
    model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            token_type_ids = d['token_type_ids'].to(device)
            labels = d['labels'].to(device)
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = loss_fn(logits, labels)
                losses.append(loss.item())
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            
    return np.mean(losses), correct_predictions / n_examples

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories if they don't exist
    os.makedirs('BERT_model/Sentiment_Analysis/models/saved', exist_ok=True)
    os.makedirs('BERT_model/Sentiment_Analysis/outputs/figures', exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract config parameters
    MODEL_NAME = config['model']['name']
    MAX_LEN = config['model']['max_len']
    N_CLASSES = config['model']['n_classes']
    BATCH_SIZE = config['training']['batch_size']
    EPOCHS = config['training']['epochs']
    PATIENCE = config['training']['patience']
    USE_AMP = config['training'].get('use_amp', False)
    N_WORKERS = config['training'].get('num_workers', 0)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    sentiment_data = SentimentData()
    df = sentiment_data.get_data()
    df = df.groupby('polarity').sample(n=9500, random_state=config['data']['random_state'])
    df = df.sample(frac=1, random_state=config['data']['random_state']).reset_index(drop=True)
    
    # Preprocessing and Data Splitting
    bert_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    df_train, df_val = train_test_split(
        df,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        shuffle=True,
        stratify=df['polarity']
    )
    
    print(f"Train set size: {len(df_train)}, Validation set size: {len(df_val)}")
    
    # Create DataLoaders
    train_data_loader = get_dataloader(
        df=df_train, 
        tokenizer=bert_tokenizer, 
        max_length=MAX_LEN, 
        batch_size=BATCH_SIZE, 
        num_workers=N_WORKERS
    )
    
    val_data_loader = get_dataloader(
        df=df_val, 
        tokenizer=bert_tokenizer, 
        max_length=MAX_LEN, 
        batch_size=BATCH_SIZE, 
        num_workers=N_WORKERS
    )
    
    # Initialize model
    model = AmazonReviewSentimentClassifier(N_CLASSES, MODEL_NAME).to(device)
    
    # Setup optimizer with different learning rates for BERT and classifier
    optimizer = torch.optim.AdamW(
        [
            {'params': model.bert_model.parameters(), 'lr': float(config['training']['learning_rate_bert'])},
            {'params': model.classifier.parameters(), 'lr': float(config['training']['learning_rate_classifier'])}
        ],
        eps=float(config['training']['epsilon']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Setup schedulers
    total_steps = len(train_data_loader) * EPOCHS
    warmup_steps = int(float(config['training']['warmup_ratio']) * total_steps)
   
    warmup_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    plateau_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.1, 
        patience=config['training']['patience']
    )
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # Training Loop
    history = defaultdict(list)
    best_accuracy = 0.0
    best_val_loss = float('inf')
    num_patience = PATIENCE
    current_patience = 0
    
    print("Training is underway... please stay tuned!")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS} [Training]")
        train_loss, train_acc = train_model(
            model=model,
            data_loader=train_data_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            warm_scheduler=warmup_scheduler,
            n_examples=len(df_train),
            device=device,
            use_amp=USE_AMP
        )
    
        print(f"\nEpoch {epoch+1}/{EPOCHS} [Validation]")
        val_loss, val_acc = eval_model(
            model=model,
            data_loader=val_data_loader,
            loss_fn=loss_fn,
            device=device,
            n_examples=len(df_val),
            use_amp=USE_AMP
        )
    
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
    
        if val_acc > best_accuracy:
            model_path = f"BERT_model/Sentiment_Analysis/models/saved/model_state_{timestamp}_no_title_acc{val_acc:.4f}.bin"
            torch.save(model.state_dict(), model_path)
            torch.save(model.state_dict(), 'BERT_model/Sentiment_Analysis/models/saved/best_model_state_no_title.bin')
            best_accuracy = val_acc
            print(f"New best model saved to {model_path} with accuracy: {best_accuracy:.4f}")
            current_patience = 0
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            current_patience = 0
        else:
            current_patience += 1
            print(f"Patience: {current_patience}/{num_patience}")
            if current_patience >= num_patience:
                print(f"Early stopping at epoch: {epoch+1}")
                break
    
        plateau_scheduler.step(val_acc)  # Monitor validation accuracy for LR plateau
        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
    
    # Save training history
    hist_df = pd.DataFrame(history)
    hist_df.index = hist_df.index + 1
    hist_df.index.name = 'epoch'
    history_path = f"BERT_model/Sentiment_Analysis/outputs/logs/training_history_{timestamp}.csv"
    hist_df.to_csv(history_path, index=True, index_label="epoch")
    print(f"Training history saved to {history_path}")
    
    # Plot training history
    figure_path = f"BERT_model/Sentiment_Analysis/outputs/figures/training_history_{timestamp}.png"
    plot_training_history(history, save_path=figure_path)
    print(f"Training plot saved to {figure_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/config.yaml'), help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)