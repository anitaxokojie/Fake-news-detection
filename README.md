# Fake News Detection: A Hybrid Approach

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This project implements a hybrid machine learning approach to detect fake news by combining linguistic features with transformer models. The approach achieves high accuracy while maintaining interpretability.

## Overview

Fake news detection remains a significant challenge in today's information ecosystem. This project addresses this challenge through:

1. Comprehensive linguistic analysis of news content
2. Semantic understanding using transformer models
3. Novel feature fusion techniques
4. Explainable predictions for transparency

## Key Features

- **Linguistic Analysis**: Extracts readability metrics, sentiment scores, and lexical diversity features from news text
- **Transformer Integration**: Utilizes DistilBERT for semantic understanding
- **Hybrid Architecture**: Combines linguistic patterns with semantic understanding
- **Multiple Models**: Includes baseline, transformer-only, enhanced hybrid, and XGBoost implementations
- **Explainability**: Provides feature importance and LIME explanations
- **Subject-Specific Analysis**: Analyzes detection performance across different news categories

## Models and Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| Bag-of-Words + RF | 92.54% | Baseline model |
| DistilBERT | 91.18% | Transformer-only model |
| Enhanced Hybrid | 94.03% | Combined linguistic and transformer features |
| XGBoost | 99.11% | Gradient boosting with TF-IDF features |

## Installation

## Quick Start
```bash
# Clone the repository
git clone https://github.com/anitaxokojie/fake-news-detection.git

# Install dependencies
pip install -r requirements.txt

# Download dataset (see data/README.md for instructions)

# Run the analysis
jupyter notebook fake_news_analysis.ipynb
