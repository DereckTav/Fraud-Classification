import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

class sentimentDataset(Dataset):
    def __init__(self, titles, reviews, labels, tokenizer, max_length=512, include_text=False):  # renamed
        self.titles = titles
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_text = include_text

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        title = str(self.titles[item])
        review = str(self.reviews[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text=title,
            text_pair=review,
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

        if self.include_text:
            output['title'] = title
            output['review'] = review

        return output

def create_custom_collate_fn(tokenizer):
    def custom_collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['label'] for item in batch]

        # Use tokenizer to pad
        encoding = tokenizer.pad(
            {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            },
            return_tensors="pt",
        )

        output = {
            "input_ids": encoding["input_ids"],
            "token_type_ids": encoding["token_type_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": torch.tensor(labels, dtype=torch.long)
        }

        # Include raw text if present
        if "title" in batch[0]:
            output["title"] = [item["title"] for item in batch]
            output["review"] = [item["review"] for item in batch]

        return output

    return custom_collate_fn

def get_dataloader(df, tokenizer, batch_size=16, max_length=512, include_text=False, num_workers=0):
    ds = sentimentDataset(
        titles=df['title'].to_numpy(),
        reviews=df['review'].to_numpy(),
        labels=df['polarity'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length,
        include_text=include_text
    )

    collate_fn = (
        create_custom_collate_fn(tokenizer)
        if include_text
        else DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', max_length=max_length)
    )

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)