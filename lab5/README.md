# Lab 5: CNN on Text Data (IMDb Sentiment Analysis)

Notebook: `DL_ass5.ipynb`

## Topics Covered

Applying CNN architectures to NLP — sentiment classification on movie reviews:

- Text preprocessing: tokenization, cleaning, vocabulary building
- Pretrained word embeddings: Word2Vec and GloVe
- CNN text encoders with multiple kernel sizes
- Attention-like pooling over time dimension
- Dropout and L1/L2 regularization for text models
- Comparative analysis of embedding quality and model capacity

## Dataset

IMDb Movie Reviews:
- 50,000 reviews (positive/negative binary labels)
- Max sequence length: ~200 tokens
- Vocabulary: ~88K words (using most common words)

## Models Implemented

| Model | Filters | Kernel Sizes | Dropout | Regularization | Embedding |
|-------|---------|--------------|---------|-----------------|-----------|
| **1** | 50 (2 kernels) | 3, 4 | ❌ | None | Word2Vec |
| **2** | 100 (2 kernels) | 3, 4 | ❌ | None | Word2Vec |
| **3** | 100 (3 kernels) | 3, 4, 5 | ✓ (0.5) | None | Word2Vec |
| **4** | 100 (3 kernels) | 3, 4, 5 | ✓ (0.5) | L2 (1e-4) | Word2Vec |
| **5** | 100 (3 kernels) | 3, 4, 5 | ✓ (0.5) | L2 (1e-4) | GloVe |

## Key Findings

**Model 1 vs Model 2** (More filters):
- Larger capacity doesn't guarantee better validation performance
- Shows that model size alone is not the limiting factor

**Model 2 vs Model 3** (Dropout + extra kernel):
- Dropout stabilizes validation loss, prevents sharp increases
- Third kernel (5-gram) captures longer semantic patterns
- Validation loss more stable: 0.24–0.26 vs 0.24–0.35

**Model 3 vs Model 4** (L2 regularization):
- L2 penalty reduces overfitting further
- Smoother gap between train and validation loss
- Training loss takes longer to drop but generalizes better

**Model 4 vs Model 5** (Word2Vec vs GloVe):
- **GloVe converges much faster** in early epochs
- GloVe learns from global co-occurrence statistics
- Word2Vec learns from local context windows
- Global semantics better suited for document-level tasks

## Dependencies

- torch
- pandas
- numpy
- scikit-learn
- gensim
- matplotlib

## Running the Notebook

```bash
jupyter notebook DL_ass5.ipynb
```

Auto-downloads IMDb dataset and pretrained embeddings. Trains 5 model variants sequentially, logs all metrics, and plots comparison curves.

## Notes

- Input sequences padded/truncated to 200 tokens
- Batch size: 128
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-Entropy
- Epochs: 3 (sufficient for convergence on this task)
