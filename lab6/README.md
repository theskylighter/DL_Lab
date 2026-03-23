# Lab 6: Sequence Models for Text Classification (IMDb Sentiment Analysis)

Notebook: `ass6.ipynb`

## Topics Covered

Implementing and comparing recurrent neural network architectures for NLP:

- **RNN (Vanilla Recurrent Neural Network)** — simplest sequence model with internal recurrence
- **LSTM (Long Short-Term Memory)** — gates to mitigate vanishing gradients
- **Bidirectional LSTM** — processes sequences in both directions
- **Packed padded sequences** — efficient handling of variable-length sequences
- **Pretrained embeddings** — initializing with Word2Vec, fine-tuning during training
- **Gradient tracking** — monitoring first-layer gradient flow during training
- **Regularization** — dropout on embeddings and hidden states

## Dataset

IMDb Movie Reviews:
- 50,000 sentiment-labeled reviews (positive/negative binary)
- Vocabulary size: ~88K words
- Max sequence length: 200 tokens (fixed padding)
- Train: 36K, Validation: 4K, Test: 10K

## Models Implemented

| Model | Architecture | Embedding | Key Feature |
|-------|--------------|-----------|-------------|
| **1** | RNN/LSTM/BiLSTM + Max Pooling | Word2Vec (300-d) | Baseline, max aggregates over time |
| **2** | RNN/LSTM/BiLSTM + Packed Sequences | Word2Vec (300-d) | Industry standard, ignores padding tokens |
| **3** | RNN/LSTM/BiLSTM + Packed + Dropout | Word2Vec (300-d) | Adds 0.5 dropout after embedding and hidden |

## Key Implementation Details

- **Hidden dimension:** 128 units
- **Optimizer:** Adam (lr=0.001)
- **Loss function:** Binary Cross-Entropy (BCELoss)
- **Epochs:** 3 (sufficient for IMDb)
- **Batch size:** 128
- **Sequence handling:** `pack_padded_sequence` for padding-aware computation

## Model Progression & Observations

**Model 1 → Model 2** (Max Pooling → Packed Sequences):
- Packed sequences allow RNN to ignore padding tokens completely
- More accurate gradient flow through valid tokens only
- Bidirectional models properly concat forward & backward hidden states

**Model 2 → Model 3** (Packed + Dropout):
- Dropout(0.5) applied after embedding and before FC classifier
- Validation curves more stable
- Reduces sharp overfitting spikes seen in Model 2

## Tracked Metrics

1. **Training loss** → logged per batch
2. **Validation accuracy** → evaluated every 100 steps
3. **Avg gradient magnitude (first layer)** → embedding layer gradients

All three metrics are visualized to compare RNN vs LSTM vs BiLSTM performance.

## Expected Results

- **RNN:** Fastest training, but gradient magnitude may decrease sharply (vanishing gradient)
- **LSTM:** Slower convergence than RNN but more stable gradients
- **BiLSTM:** Highest accuracy; strongest validation performance (uses more parameters)

Dropout and packed sequences should make validation curves smoother than Model 1.

## Dependencies

- torch
- pandas
- numpy
- scikit-learn
- gensim
- matplotlib

## Running the Notebook

```bash
jupyter notebook ass6.ipynb
```

Auto-downloads IMDB dataset and Word2Vec embeddings. Trains all 3 model variants (RNN, LSTM, BiLSTM) across 3 architecture types (Model 1/2/3). Logs and plots comparative metrics.

## Notes

- Word2Vec vectors auto-downloaded from gensim (first run may take 2–3 min)
- Dataset auto-downloaded from GitHub (needs internet on first run)
- GPU recommended (auto-detected)
- Gradient magnitudes reset per batch; values shown are mean over the batch
