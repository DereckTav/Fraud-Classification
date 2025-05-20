import pandas as pd
import re
import gc
import os

class reviewData:
    def __init__(self, type='train', chunk_size=10000):
        if type not in ['train', 'test']:
            raise ValueError("Invalid 'type'. Only 'train' or 'test' are allowed.")
        
        if type in ['train']:
            file_path='Datasets/amazon_review_full_csv/train.csv'
        else:
            file_path='Datasets/amazon_review_full_csv/test.csv'

        # Use chunksize to read large files in manageable portions
        self.chunk_size = chunk_size
        self.file_path = file_path
        self.data = pd.DataFrame()

        self.output_path = f'Datasets/processed_datasets/amazon_review_full_csv/normalized-data/{type}.csv'
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        if not os.path.exists(self.output_path):
            self.process_data()
        else:
            self.data = pd.read_csv(self.output_path)

    def get_data(self):
        return self.data

    def process_data(self):
        # Process the data in chunks
        for chunk in pd.read_csv(self.file_path, header=None, names=['rating', 'title', 'review'], chunksize=self.chunk_size):
            chunk['title'] = chunk['title'].apply(self.preprocess)
            chunk['review'] = chunk['review'].apply(self.preprocess)

            # Append the processed chunk to the main dataframe
            self.data = pd.concat([self.data, chunk], ignore_index=True)

            del chunk
            gc.collect()

        # save the processed data to a new file
        self.data.to_csv(self.output_path, index=False)

    def preprocess(self, text):
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return "NULL"

        text = re.sub(r'https?://\S+|www\.\S+', '', text) # remove urls
        text = re.sub(r'\S+@\S+', '', text) # remove emails
        text = re.sub(r'\s+', ' ', text).strip() # normalize spaces

        return self.normalize_case(text) or "NULL"

    def normalize_case(self, text):
        words = text.split()
        for i, word in enumerate(words):
            # Leave all-uppercase words as is
            if word.isupper() and len(word) > 1:
                continue

            # If the word has more than 1 capital letter, leave it as is
            if sum(1 for c in word if c.isupper()) > 1:
                continue

            words[i] = word.lower()

        return ' '.join(words)

if __name__ == '__main__':
    # make train data
    data = reviewData()

    print(data.get_data()['rating'].value_counts())
    check_nan = data.get_data()['title'].isnull().values.any()
    print(check_nan)
    print(data.get_data()[data.get_data()['title'].isnull()])

    # make test data
    reviewData(type='test')
