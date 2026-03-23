# Lab 3: Optimizer Comparisons

Notebook: `DL03_optimizers.ipynb`

## Topics Covered

Comparative analysis of different gradient-based optimizers for training neural networks:

- **Gradient Descent (GD)** — vanilla approach, updates with constant learning rate
- **Stochastic Gradient Descent (SGD)** — updates on single samples (noisier, sometimes faster escape from local minima)
- **Momentum** — accumulates velocity to smooth updates
- **Nesterov Accelerated Gradient (NAG)** — "look-ahead" variant of momentum
- **AdaGrad** — adapts learning rate per parameter based on historical gradient magnitudes
- **RMSProp** — adaptive learning rate with exponential moving average
- **Adam** — combines momentum and adaptive learning rate (Adaptive Moment Estimation)

## Experiment Setup

- **Task:** 2-layer neural network for regression
- **Dataset:** Toy 1D dataset (200 samples, linear relationship + noise)
- **Metric:** MSE loss over training iterations
- **Observation:** Convergence speed, stability, and final loss for each optimizer

## Key Insights Expected

1. Adam and RMSProp converge faster than vanilla GD
2. Momentum helps escape plateaus
3. NAG typically performs better than momentum alone
4. SGD is noisier but useful for escaping sharp minima

## Dependencies

- numpy
- matplotlib

## Running the Notebook

```bash
jupyter notebook DL03_optimizers.ipynb
```

Trains the same network architecture with each optimizer and compares loss curves visually.
