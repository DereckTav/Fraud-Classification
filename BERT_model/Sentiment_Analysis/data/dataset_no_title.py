import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

class ReviewOnlySentimentDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=512):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text=review,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        output = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        return output
    
def get_dataloader(df, tokenizer, batch_size=16, max_length=512, num_workers=0):
    ds_review_only = ReviewOnlySentimentDataset(
        reviews=df['review'].to_numpy(),
        labels=df['polarity'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', max_length=max_length)
    
    return DataLoader(ds_review_only, batch_size=batch_size, collate_fn=collator, shuffle=False, num_workers=num_workers, pin_memory=True)