# Lab 1: Introduction to PyTorch Tensors

Notebook: `dl_01.ipynb`

## Topics Covered

Fundamental PyTorch tensor operations and properties:

- **Creation:** scalars, vectors, matrices; various dtypes
- **Properties:** shape, ndim, dtype, device (CPU/GPU)
- **Special tensors:** zeros, ones, random, identity
- **Type conversion:** .float(), .int(), .to(dtype)
- **Arithmetic:** addition, subtraction, multiplication, division
- **Statistical ops:** mean, std, sum along dimensions
- **Reshaping:** view(), reshape(), squeeze(), unsqueeze()

## Key Concepts

1. **Rank/Dimension** — how many indices needed to address an element
2. **Shape** — size along each dimension
3. **Data type** — float32, int64, etc.
4. **Device** — CPU or GPU
5. **Immutability vs In-place** — operations create new tensors unless using `_` suffix

## Learning Objectives

By the end of this lab, you should be able to:
- Create and inspect tensors
- Convert data types and move tensors between devices
- Perform basic tensor manipulation for preprocessing
- Understand broadcasting behavior

## Dependencies

- torch

## Running the Notebook

```bash
jupyter notebook dl_01.ipynb
```

Each cell demonstrates a key concept with executable examples.

