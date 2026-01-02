# Hybrid NLP News Verification Pipeline
### Multi-Modal Fake News Detection Using Semantic + Stylistic Fusion

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Why This Matters

In the modern information ecosystem, disinformation is a multi-billion dollar problem affecting brand safety, market stability, and public trust.

**The Problem:** Purely semantic models (like standard BERT) often focus on *what is said*, missing the subtle stylistic "tells" of deceptive writing.

**The Solution:** I engineered a hybrid pipeline that fuses **Contextual Semantic Embeddings** with **Handcrafted Stylistic Features** (Readability, Sentiment, and Lexical Variety).

**The Result:** By analyzing both the *content* and the *writing style*, this system provides a robust framework for automated content moderation that is not only accurate but also explainable via transparent decision paths.

---

## 2. Quick Results

I benchmarked four distinct architectures to identify the optimal balance between accuracy and inference latency.

| Method | Accuracy | F1 Score | Status |
|--------|----------|----------|--------|
| Random Forest (Baseline) | 92.54% | 0.93 | Statistical Baseline |
| DistilBERT (Semantic) | 89.71% | 0.90 | Contextual Core |
| Hybrid Fusion (This Project) | 64.00%* | 0.59 | Neural Proof-of-Concept |
| Optimized XGBoost | 100.00% | 1.00 | Diagnostic Champion |

**Business Impact:** While XGBoost achieved 100% accuracy, my diagnostic analysis revealed data leakage (source tags). In a production environment, the **Hybrid Fusion Model** is the preferred architecture as it is designed to generalize across novel topics by balancing semantics with latent stylistic markers.

### Performance Visualizations

![Training Curves](processed/models/enhanced_model/training_curves.png)

*Figure 1: (Left) Loss convergence showing the "Cold Start" challenge of deep fusion. (Right) Accuracy scaling across training epochs.*

---

## 3. Code Architecture

### Core Pipeline Components

#### 1. Multi-Modal Feature Extraction
```python
def extract_linguistic_features(text):
    """Calculates 10+ stylistic signals including Readability & Sentiment"""
    stats = [
        textstat.flesch_kincaid_grade(text),
        sentiment_analyzer.polarity_scores(text)['compound'],
        len(set(text.split())) / len(text.split())  # Lexical Diversity
    ]
    return stats
```

#### 2. Custom Neural Fusion Head (PyTorch)
```python
class EnhancedNewsClassifier(nn.Module):
    """Fuses BERT embeddings with Stylistic vectors using Gating"""
    def forward(self, input_ids, attention_mask, linguistic_features):
        transformer_out = self.transformer(input_ids, attention_mask)
        hidden_state = transformer_out.last_hidden_state[:, 0, :]
        
        # Gating Mechanism: deciding signal influence
        gate_values = self.gate(linguistic_features)
        fused = hidden_state * gate_values
        return self.classifier(fused)
```

#### 3. Explainable AI (XAI) Layer
```python
# Mapping black-box decisions to human-readable tokens
exp = explainer.explain_instance(sample_text, model.predict_proba)
exp.show_in_notebook()
```

---

## 4. How It Works

The pipeline follows a modular R&D process:

1. **Normalization:** Regex-based cleaning that preserves punctuation signatures used in stylistic analysis.
2. **Semantic Encoding:** Fine-tuning a `distilbert-base-uncased` transformer.
3. **Stylistic Encoding:** Parallel extraction of 10 handcrafted linguistic metrics.
4. **Feature Fusion:** Using a custom PyTorch head to evaluate Concatenation, Attention, and Gated integration strategies.
5. **Interpretation:** Using LIME to visualize which specific words (e.g., factual attribution vs. emotional baiting) drove the prediction.

---

## 5. Key Technical Decisions

### Why DistilBERT over standard BERT?

In production content moderation, **Inference Latency** is as important as accuracy. DistilBERT provides 97% of BERT's performance while being 40% smaller and 60% faster.

### The "Reuters" Discovery (Data Integrity)

During training, XGBoost hit 100% accuracy. By analyzing Feature Importance, I discovered that the model identified "Reuters" tags as a proxy for truth.

**Decision:** I chose to document this as a diagnostic finding. It proves the model's sensitivity and highlights the need for adversarial filtering (stripping source signatures) in future iterations.

---

## 6. Architecture: Multi-Modal Neural Fusion

The core innovation of this project is the `EnhancedNewsClassifier`. Unlike simple concatenation, I implemented a **Gating Mechanism**:

- **Signal Filtering:** The model learns a sigmoid-activated gate that controls how much the linguistic features should "nudge" the semantic decision.
- **Why?** This prevents linguistic noise from over-riding strong semantic signals, making the model more robust than simple linear concatenation.

---

## 7. Project Structure
```
news-verification-pipeline/
├── Hybrid_NLP_News_Verification_Pipeline.ipynb  # Main R&D Pipeline
├── requirements.txt                             # Dependency manifests
├── README.md                                    # Documentation
├── processed/
│   ├── models/                                  # Trained weights (.pth, .pkl)
│   └── data/                                    # Stratified train/test splits
└── output/                                      # Visualizations & LIME Reports
```

---

## 8. What I Learned

**Deep Learning Constraints:** The Hybrid model's 64% accuracy (vs. 92% baseline) demonstrates the "Cold Start" problem. Complex neural fusion requires significantly more data (N > 5,000) to outperform traditional ensemble methods.

**Explainability is a Requirement:** Accuracy is not enough for legal or social media compliance. Integrating LIME allowed me to verify that the model was looking at "attribution" words rather than "stop words."

---

## 9. Getting Started

### Prerequisites

Clone this repository:
```bash
git clone https://github.com/anitaxokojie/fake-news-detection.git
cd fake-news-detection
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the Pipeline

Open the notebook to view the end-to-end data engineering and model benchmarking:
```bash
jupyter notebook Hybrid_NLP_News_Verification_Pipeline.ipynb
```

---

## Project Context

- **Type:** Technical Research & Development
- **Affiliation:** University of London, Goldsmiths (Advanced Machine Learning Initiative)
- **Focus:** Hybrid NLP Architectures & Model Interpretability

---

## Future Roadmap

- **Adversarial Training:** Implement a "source-blind" cleaning layer to remove metadata-based leakage.
- **Scaling:** Deploy the Gated Fusion model to the full 40,000-row dataset using a GPU-accelerated cluster to overcome sample constraints.

---

## License

MIT License

---

## Contact

**Anita Okojie**  
[LinkedIn](https://linkedin.com/in/anitaxokojie) | [Email](mailto:anitaxokojie@gmail.com)

Built with ☕ and PyTorch.
