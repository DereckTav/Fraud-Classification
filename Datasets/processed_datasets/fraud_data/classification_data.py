import pandas as pd
import re
import gc
import os

class classificationData:
    def __init__(self, type='train', chunk_size=10000):
        if type not in ['train']:
            raise ValueError("Invalid 'type'. Only 'train' is allowed.")
        
        if type in ['train']:
            file_path='Datasets/fraud_data/train.csv'

        # Use chunksize to read large files in manageable portions
        self.chunk_size = chunk_size
        self.file_path = file_path
        self.data = pd.DataFrame()

        #Note if dataset already exists you do not need to provide file_path
        #Can just provide type.
        self.output_path = f'Datasets/processed_datasets/fraud_data/data/{type}_data.csv'
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        if not os.path.exists(self.output_path):
            self.process_data()

        else:
            self.data = pd.read_csv(self.output_path)

    def get_data(self):
        return self.data

    def process_data(self):
        # Process the data in chunks
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            chunk['Fraud'] = chunk['Fraud']
            chunk['rating'] = chunk['rating']
            chunk['review'] = chunk['review'].apply(self.preprocess)

            # Append the processed chunk to the main dataframe
            self.data = pd.concat([self.data, chunk], ignore_index=True)

            del chunk
            gc.collect()

        # save the processed data to a new file
        self.data.to_csv(self.output_path, index=False)

    def preprocess(self, text):

        text = re.sub(r'https?://\S+|www\.\S+', '', text) # remove urls
        text = re.sub(r'\S+@\S+', '', text) # remove emails
        text = re.sub(r'\s+', ' ', text).strip() # normalize spaces

        return text

if __name__ == '__main__':
    # make train data
    data = classificationData()

    print(data.get_data().isna().any())
