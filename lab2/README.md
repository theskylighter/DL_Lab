# Lab 2: Backpropagation in Multilayer Neural Networks

Notebook: `Assignment_2.ipynb`

## Topics Covered

- Manual implementation of forward pass for logistic regression
- One-hot encoding for multi-class classification
- Backpropagation algorithm from scratch
- Gradient computation and weight updates
- Training a multi-layer neural network on the sklearn digits dataset
- Performance evaluation with accuracy metrics

## Dataset

sklearn digits (8×8 grayscale images of handwritten digits 0–9)
- 1,797 samples, 10 classes
- 64 features (flattened 8×8 pixels)

## Key Concepts

1. **Logistic Regression** — simple linear classifier with softmax activation
2. **Backpropagation** — computing gradients via the chain rule
3. **Gradient Descent** — parameter updates using computed gradients
4. **Data Preprocessing** — normalization with StandardScaler

## Dependencies

- torch
- scikit-learn
- matplotlib

## Running the Notebook

```bash
jupyter notebook Assignment_2.ipynb
```

Preprocesses the digits dataset, implements forward and backward passes, and trains the model end-to-end.
