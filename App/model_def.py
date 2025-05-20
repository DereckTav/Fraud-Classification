import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
import logging
import os

logger = logging.getLogger(__name__)

class FraudClassifier(nn.Module):
    def __init__(self, MODEL_NAME, dropout=0.3):
        super(FraudClassifier, self).__init__()
        try:
            self.bert = BertModel.from_pretrained(MODEL_NAME)
        except Exception as e:
            try:
                self.bert = AutoModel.from_pretrained(MODEL_NAME)
                logger.info("Model loaded successfully with AutoModel")
            except Exception as e2:
                raise RuntimeError(f"Could not load model {MODEL_NAME}: {str(e2)}")
                
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        try:
            if token_type_ids is not None:
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            else:
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
            # Handle different output formats from different BERT variants
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # Use CLS token if pooler_output not available
                pooled_output = outputs.last_hidden_state[:, 0, :]
            else:
                # Fall back to first element if outputs is a tuple
                pooled_output = outputs[0][:, 0, :]
                
            x = self.dropout(pooled_output)
            return self.linear(x)
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise