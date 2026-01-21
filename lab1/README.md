# Deep Learning Lab 1: Introduction to PyTorch Tensors

This directory contains a Jupyter Notebook (`dl_01.ipynb`) that serves as an introduction to fundamental PyTorch tensor operations.

## `dl_01.ipynb` Overview

The notebook covers the following key concepts:

- **Scalar, Vector, and Matrix Creation**: Demonstrates how to create 0-dimensional (scalar), 1-dimensional (vector), and 2-dimensional (matrix) tensors.
- **Tensor Properties**: Explores attributes of tensors such as `ndim` (number of dimensions), `item()` (getting Python number from a scalar tensor), `shape`, `dtype` (data type), and `device`.
- **Special Tensor Creation**: Shows how to create tensors filled with zeros (`torch.zeros`), ones (`torch.ones`), and random values (`torch.rand`).
- **Data Type Manipulation**: Illustrates how to change the data type of a tensor (e.g., using `.float()` or `.to(torch.float32)`).
- **Tensor Operations**: Covers basic arithmetic operations (addition, subtraction, multiplication) and statistical operations (mean, standard deviation, sum along a dimension).
- **Tensor Reshaping**: Explains how to change the shape of tensors using `view()` and `reshape()`.

This notebook is a foundational resource for understanding how to work with tensors in PyTorch, which is crucial for building deep learning models.
