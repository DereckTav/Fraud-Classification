import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from functools import partial

class classifyDataset(Dataset):
    def __init__(self, titles, reviews, ratings, tokenizer, max_length=512, include_text=False):
        self.titles = titles
        self.reviews = reviews
        self.ratings = ratings  # Kept for metadata/logging
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_text = include_text

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        title = str(self.titles[item])
        review = str(self.reviews[item])
        rating = self.ratings[item]

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
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

        if self.include_text:
            output['rating'] = rating
            output['title'] = title
            output['review'] = review

        return output

# Define this at the top level, NOT inside another function
def custom_collate_fn(batch, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    encoding = tokenizer.pad(
        {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        },
        return_tensors="pt"
    )

    output = {
        "input_ids": encoding["input_ids"],
        "token_type_ids": encoding["token_type_ids"],
        "attention_mask": encoding["attention_mask"],
    }

    if "title" in batch[0]:
        output["review"] = [item["review"] for item in batch]
        output["title"] = [item["title"] for item in batch]

    if "rating" in batch[0]:
        output["rating"] = [item["rating"] for item in batch]

    return output


def get_dataloader(df, tokenizer, batch_size=16, max_length=512, include_text=False, num_workers=0):
    ds = classifyDataset(
        titles=df['title'].astype(str).to_numpy(),
        reviews=df['review'].astype(str).to_numpy(),
        ratings=df['rating'].to_numpy(),  # Ignored by model
        tokenizer=tokenizer,
        max_length=max_length,
        include_text=include_text
    )

    collate_fn = (
        partial(custom_collate_fn, tokenizer=tokenizer)
        if include_text
        else DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', max_length=max_length)
    )

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=num_workers, pin_memory=True)
