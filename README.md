# Deep Learning Lab Assignments

Comprehensive collection of implementations across foundational deep learning topics: tensors, backpropagation, optimization, CNNs, text classification, and sequence models.

## Repository Structure

```
lab1/  → PyTorch tensor fundamentals
lab2/  → Backpropagation & gradient descent
lab3/  → Optimizer comparisons
lab4/  → CNN on CIFAR-10
lab5/  → CNN on text (IMDb sentiment)
lab6/  → RNN/LSTM/BiLSTM on text (IMDb sentiment)
```

---

## Completed Labs


<details open>
<summary><strong>🔬 Lab 1: Introduction to PyTorch Tensors</strong></summary>

**Notebook:** `lab1/dl_01.ipynb`

Fundamentals of PyTorch tensor operations:
- Tensor creation: scalars, vectors, matrices
- Tensor properties: shape, dtype, device, ndim
- Special tensors: zeros, ones, random
- Type conversion and reshaping
- Arithmetic and statistical operations

**Dependencies:** `torch`

</details>

---

<details>
<summary><strong>📐 Lab 2: Backpropagation in Multilayer Neural Networks</strong></summary>

**Notebook:** `lab2/Assignment_2.ipynb`

Implementing gradient descent and backpropagation from scratch:
- Logistic regression with manual gradient computation
- Multi-layer neural network classifier
- Training on sklearn digits dataset
- Performance metrics and accuracy evaluation

**Dependencies:** `torch`, `scikit-learn`, `matplotlib`

</details>

---

<details>
<summary><strong>⚙️ Lab 3: Optimizer Comparisons</strong></summary>

**Notebook:** `lab3/DL03_optimizers.ipynb`

Comparative analysis of gradient-based optimizers:
- Standard gradient descent (GD)
- Stochastic gradient descent (SGD)
- Momentum
- Nesterov accelerated gradient (NAG)
- AdaGrad
- RMSProp
- Adam

Training a 2-layer neural network on a toy regression dataset to observe convergence behavior.

**Dependencies:** `numpy`, `matplotlib`

</details>

---

<details>
<summary><strong>🖼️ Lab 4: Convolutional Neural Networks (CIFAR-10)</strong></summary>

**Notebook:** `lab4/CNN.ipynb`

Building and training CNNs for image classification:
- 2-layer CNN + 2 fully-connected layers
- CIFAR-10 dataset (10-class image classification)
- Training, validation, and test evaluation
- L1/L2 regularization effects
- Hyperparameter experiments

**Dependencies:** `torch`, `torchvision`, `matplotlib`, `numpy`

</details>

---

<details>
<summary><strong>📝 Lab 5: CNN on Text Data (IMDb Sentiment Analysis)</strong></summary>

**Notebook:** `lab5/DL_ass5.ipynb`

Applying CNNs to text classification using pretrained embeddings:
- Data: IMDb movie reviews dataset
- Multiple CNN variants with 2–3 convolution layers
- Pretrained embeddings: Word2Vec and GloVe
- Regularization techniques: dropout, L1, L2
- Observations on filter count, regularization effects, and embedding choice

**5 Models Compared:**
1. Basic CNN (50 filters per kernel size)
2. Larger CNN (100 filters per kernel size)
3. CNN with 3 kernel sizes + dropout
4. Model 3 + L2 regularization
5. Model 3/4 architecture with GloVe embeddings

**Key Findings:** GloVe embeddings converge faster; dropout & L2 reduce overfitting; larger capacity doesn't guarantee better generalization.

**Dependencies:** `torch`, `pandas`, `numpy`, `scikit-learn`, `gensim`, `matplotlib`

</details>

---

<details>
<summary><strong>🔄 Lab 6: Sequence Models for Text (IMDb Sentiment Analysis)</strong></summary>

**Notebook:** `lab6/ass6.ipynb`

Implementing and comparing RNN/LSTM/BiLSTM for sentiment classification:
- Data: IMDb movie reviews (same as Lab 5)
- Pretrained Word2Vec embeddings (300-dim)
- Text encoding with vocabulary and padding (max_length=200)

**3 Model Architectures:**
1. **Model 1:** Baseline with max pooling over time
2. **Model 2:** Pack padded sequences (ignore padding tokens)
3. **Model 3:** Model 2 + dropout regularization

**Each variant tested with:**
- RNN, LSTM, and Bidirectional-LSTM

**Tracked Metrics:**
- Training loss
- Validation accuracy
- Average gradient magnitude (first layer)

**Key Observations:**
- Packed sequences improve training by ignoring padding
- Bidirectional models capture broader context
- Dropout helps stabilize validation curves

**Dependencies:** `torch`, `pandas`, `numpy`, `scikit-learn`, `gensim`, `matplotlib`

</details>

---

## Environment Setup

<details>
<summary><strong>Install Dependencies</strong></summary>

```bash
pip install -r requirements.txt
```

</details>

<details>
<summary><strong>Run a Lab</strong></summary>

Open any notebook in Jupyter:
```bash
jupyter notebook lab1/dl_01.ipynb
```

</details>

<details>
<summary><strong>Hardware</strong></summary>

GPU is optional but recommended for Labs 4, 5, 6. Code auto-detects CUDA availability.

</details>

---

## Notes

- Notebooks may contain cached outputs from previous runs
- All labs use random seeds for reproducibility where applicable
- Datasets (CIFAR-10, IMDb) are auto-downloaded on first run (may require internet)
- Lab 5 and 6 both use IMDb data but with different architectures (CNN vs. RNN)
