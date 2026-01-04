from .base import BaseEncoder
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

class TextEncoder(BaseEncoder):
    def __init__(self, model_name: str = 'roberta-base', frozen: bool = True):
        super().__init__(output_dim=768)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        
        if frozen:
            self.freeze()
            
    def forward(self, texts: list[str]) -> torch.Tensor:
        # Tokenize
        encoded = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=128
        )
        
        # Move to device
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Model Forward
        outputs = self.model(input_ids, attention_mask=attention_mask)
        
        # Use CLS token representation (first token)
        # Shape: (Batch, 768)
        features = outputs.last_hidden_state[:, 0, :]
        return features
