import pandas as pd
import re
import gc
import os

class classificationData:
    def __init__(self, type='train', chunk_size=10000):
        if type not in ['train', 'test']:
            raise ValueError("Invalid 'type'. Only 'train' or 'test' are allowed.")
        
        if type in ['train']:
            file_path='Datasets/amazon_review_dataset_2018/train.csv'
        else:
            file_path='Datasets/amazon_review_dataset_2018/test.csv'

        # Use chunksize to read large files in manageable portions
        self.chunk_size = chunk_size
        self.file_path = file_path
        self.data = pd.DataFrame()

        self.output_path = f'Datasets/processed_datasets/amazon_review_dataset_2018/normalized-data/{type}.csv'
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
            chunk['review'] = chunk['review'].apply(self.preprocess)

            # Append the processed chunk to the main dataframe
            self.data = pd.concat([self.data, chunk], ignore_index=True)

            del chunk
            gc.collect()

        # save the processed data to a new file
        self.data.to_csv(self.output_path, index=False)

    def preprocess(self, text):

        text = re.sub(r'https?://\S+|www\.\S+', '', text) 
        text = re.sub(r'\S+@\S+', '', text)              
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[\r\n]+', ' ', text)  # Replace all line breaks with a single space        

        return text

if __name__ == '__main__':
    # make train data
    data = classificationData()

    print(data.get_data().isna().any())
    # make test data
    classificationData(type='test')
