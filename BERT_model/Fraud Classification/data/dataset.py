import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

class fraudDataset(Dataset):
    def __init__(self, ratings, reviews, labels, tokenizer, max_length=512, include_text=False):
        self.ratings = ratings
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_text = include_text

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        rating = str(self.ratings[item])
        review = str(self.reviews[item])
        label = int(self.labels[item])

        encoding = self.tokenizer(
            text=rating,
            text_pair=review,
            padding=False,           # Let the collator handle padding
            truncation=True,
            max_length=self.max_length,
            return_tensors=None      # Return plain lists
        )

        output = {
            'input_ids': encoding['input_ids'],
            'token_type_ids': encoding.get('token_type_ids', None),
            'attention_mask': encoding['attention_mask'],
            'label': label           # Don't wrap with torch.tensor here; let collator or trainer handle it
        }

        if self.include_text:
            output['rating'] = rating
            output['review'] = review
            output['Fraud'] = label

        return output

def get_dataloader(df, tokenizer, batch_size=16, max_length=512, include_text=False, num_workers=0):
    ds = fraudDataset(
        ratings=df['rating'].to_numpy(),
        reviews=df['review'].to_numpy(),
        labels=df['Fraud'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length,
        include_text=include_text
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', max_length=max_length)

    return DataLoader(
        ds, 
        batch_size=batch_size, 
        collate_fn=collator, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )