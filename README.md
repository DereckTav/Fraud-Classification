# Review Fraud Detection System

## Overview
This project implements a machine learning system that classifies product reviews as either fraudulent or non-fraudulent. The goal is to help consumers identify authentic reviews that provide reliable information about products they're considering purchasing.

The system uses a two-stage approach:
1. **Sentiment Classification**: Determines whether a review is positive or negative
2. **Fraud Detection**: Identifies reviews that are likely to be fraudulent (computer-generated)

## Key Features
- BERT-based deep learning models fine-tuned for review classification
- Custom data loaders and preprocessing pipelines
- Dual model approach (with/without review titles)
- CLI tool for testing fraud detection on custom reviews
- Comprehensive evaluation metrics and visualization

## Project Structure
```
Fraud Classification/
├──App/
│   ├── config/              # Holds config.yaml, which contains hyperparams, etc...
│   ├── model/               # Holds the model that is being used to classify fraud
│   ├── cli.py               # The way you will interact with the model
│   └── model_def.py  
├── BERT_model/
│   │   ├── Fraud Classification          # Training for Fraud Model
│   │   ├── Sentiment_Analysis            # Training for sentiment classification Model
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Installation

### Requirements
- Python 3.8+
- CUDA 12.8+ (optional, for GPU acceleration)

### Dependencies
The project requires the following libraries:
```
torch>=1.7.0  
transformers>=4.5.0  
pandas>=1.0.0  
numpy>=1.19.0  
scikit-learn>=0.24.0  
matplotlib>=3.3.0  
seaborn>=0.11.0  
pyyaml>=5.4.0  
huggingface_hub[hf_xet]>=0.23.0
```

### Standard Installation
```bash
pip install -r requirements.txt
```

### Manual Installation
```bash
pip install torch>=1.7.0 \
            transformers>=4.5.0 \
            pandas>=1.0.0 \
            numpy>=1.19.0 \
            scikit-learn>=0.24.0 \
            matplotlib>=3.3.0 \
            seaborn>=0.11.0 \
            pyyaml>=5.4.0 \
            'huggingface_hub[hf_xet]>=0.23.0'
```

### GPU Support
For GPU acceleration, install the CUDA-compatible version of PyTorch:
```bash
# First, install CUDA 12.8 from NVIDIA's website
# Then install the CUDA-compatible PyTorch version
```

## Usage

### Dataset Preparation
1. **Sentiment Analysis Datasets**: 
   - Download the [Amazon Review Polarity CSV dataset](https://nijianmo.github.io/amazon/index.html)
   - Place it in the `datasets/` directory corresponding to amazon_review_sentiment

2. **Fraud Classification Dataset**:
   - A sample dataset is provided in `datasets/fraud_data.csv` (which is only needed by fraud classifier training)
   - The below link is for if you want to run any of the files in processed_datasets/amazon_review_dataset_2018
     - For full reproduction, download the [2018 Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html)

### Training Models

#### Sentiment Classification
```bash
# Train the model with review titles
cd BERT_model/Sentiment_Analysis
python train.py

# Train the model without review titles
python train_no_title.py
```

#### Fraud Classification
```bash
cd BERT_model/Fraud_Classification
python train.py
```

After training, select the best model checkpoint (typically the one with the highest validation accuracy) and move it to the `model/` directory.

### Testing Fraud Detection
```bash
cd App
python cli.py
```
This launches an interactive CLI where you can enter a review text and rating to test if it's classified as fraudulent.

## Results
The current implementation achieves:
- 94.3% accuracy on sentiment classification with review titles
- 92.1% accuracy on sentiment classification without titles
- 100% validation accuracy on fraud detection (epoch 3 model)

## Future Improvements
- Increase the training dataset size for fraud classification
- Incorporate product category stratification
- Deploy as a web service with API
- Experiment with alternative model architectures
- Detect human-written fraudulent reviews (not just computer-generated)

## Note
This project is an initial proof of concept with satisfactory results. Due to computational resource constraints, models were trained on a limited dataset. The approach demonstrates viability, but would benefit from additional data and computing resources for a production-ready system.

## License
[MIT](LICENSE)
