# Lab 4: Convolutional Neural Networks (CIFAR-10)

Notebook: `CNN.ipynb`

## Topics Covered

Building and training CNNs for image classification:

- CNN architecture design with convolutional and pooling layers
- Transfer of learning with pre-trained embeddings
- Hyperparameter tuning: learning rate, batch size, number of filters
- Regularization techniques: L1, L2 penalties, dropout
- Training, validation, and test evaluation loops
- Loss curves and overfitting analysis

## Dataset

CIFAR-10 (Canadian Institute for Advanced Research):
- 60,000 32×32 RGB images
- 10 classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training + 10,000 test (80/10/10 train/val/test split)

## Model Architecture

Basic CNN:
- Conv2d(3 input channels → 6 filters, 5×5 kernel)
- MaxPool2d(2×2)
- Conv2d(6 → 16 filters, 5×5 kernel)
- MaxPool2d(2×2)
- Flatten
- Linear(16×5×5 → 120 hidden units)
- Linear(120 → 10 classes)
- Softmax output

## Key Experiments

1. **Learning rate sweep** — test different alpha values
2. **BatchNorm impact** — compare with/without batch normalization
3. **Dropout effects** — observe overfitting reduction
4. **L1/L2 regularization** — weight penalty impact on generalization

## Dependencies

- torch
- torchvision
- matplotlib
- numpy

## Running the Notebook

```bash
jupyter notebook CNN.ipynb
```

Downloads CIFAR-10 automatically, trains the CNN, logs metrics, and visualizes loss/accuracy curves.
