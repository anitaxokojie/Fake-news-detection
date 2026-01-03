import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Initialize tools
analyzer = SentimentIntensityAnalyzer()

def extract_linguistic_features(text):
    """
    Extracts the stylometric features used by the hybrid model.
    Must match the 10-feature vector used during training.
    """
    if not isinstance(text, str) or text.strip() == "":
        return [0] * 10

    # Readability
    fre = textstat.flesch_reading_ease(text)
    fkg = textstat.flesch_kincaid_grade(text)
    smog = textstat.smog_index(text)
    
    # Sentiment
    vs = analyzer.polarity_scores(text)
    
    # Complexity/Lexical
    words = text.split()
    word_count = len(words)
    sent_count = textstat.sentence_count(text)
    lexical_diversity = len(set(words)) / max(word_count, 1)

    return [
        fre, fkg, smog, 
        vs['pos'], vs['neg'], vs['neu'], vs['compound'],
        word_count, sent_count, lexical_diversity
    ]

class EnhancedNewsClassifier(nn.Module):
    """
    Custom PyTorch Model: Fuses DistilBERT embeddings with Stylometric features
    using a Gating Mechanism.
    """
    def __init__(self, pretrained_model_name="distilbert-base-uncased", num_ling_features=10):
        super(EnhancedNewsClassifier, self).__init__()
        self.transformer = DistilBertModel.from_pretrained(pretrained_model_name)
        self.transformer_dim = self.transformer.config.hidden_size
        
        # Gating Mechanism: Learns which stylistic features to prioritize
        self.gate = nn.Sequential(
            nn.Linear(num_ling_features, self.transformer_dim),
            nn.Sigmoid()
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, ling_features):
        # 1. Semantic Branch
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # 2. Stylometric Branch (Gated Fusion)
        gate_values = self.gate(ling_features)
        fused = hidden_state * gate_values
        
        # 3. Output
        return self.classifier(fused)

def predict_article(text, model_path='processed/models/enhanced_model/final_model.pt'):
    """
    Production-ready inference function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Tokenizer and Model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = EnhancedNewsClassifier()
    
    # Load trained weights (handling CPU/GPU mapping)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Preprocess Text
    inputs = tokenizer.encode_plus(
        text, 
        max_length=128, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    
    ling_feat = torch.tensor([extract_linguistic_features(text)], dtype=torch.float32)

    # Inference
    with torch.no_grad():
        logits = model(
            inputs['input_ids'].to(device), 
            inputs['attention_mask'].to(device), 
            ling_feat.to(device)
        )
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = np.argmax(probs)
    
    label_map = {0: "Real", 1: "Fake"}
    return {
        "label": label_map[prediction],
        "confidence": probs[prediction],
        "probabilities": {"Real": probs[0], "Fake": probs[1]}
    }

if __name__ == "__main__":
    # Simple CLI Test
    sample = "BREAKING: New discovery changes everything we know about the world!"
    print(f"Testing Model... \nText: {sample}")
    # Note: This requires the model file to exist to work
    # result = predict_article(sample)
    # print(result)
