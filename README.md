# Gibbs-Sampling-and-Gaussian-Mixture-Models

This repository contains the implementation of advanced Bayesian Machine Learning algorithms, focusing on **Robust Linear Regression** and **Bayesian Gaussian Mixture Models (GMMs)**. The core of this project revolves around using **Gibbs Sampling** to perform approximate inference in complex probabilistic models involving latent variables.

## 🧠 Project Overview

This project tackles two primary challenges in Machine Learning from a Bayesian perspective:
1.  **Regression with Outliers:** Standard linear regression is highly sensitive to outliers. By introducing a latent variable indicating whether a data point is an outlier, we can perform robust regression that inherently ignores noisy data points.
2.  **Probabilistic Clustering with Unknown Clusters:** Standard GMMs (via EM) require knowing the exact number of clusters beforehand and can suffer from overfitting. A Bayesian GMM places a Dirichlet prior over the mixture weights, allowing the model to automatically determine the effective number of clusters, while priors on the means and covariances regularize the model.

## ✨ Key Features & Implementations

### 1. Robust Regression via Gibbs Sampling
*   **Algorithm:** Implements a Gibbs sampling algorithm that alternates between sampling outlier indicator variables and fitting a Bayesian Linear Regression model on the inferred "inliers."
*   **1D Synthetic Data:** Demonstrates the ability of the model to completely ignore extreme outliers and find the true underlying linear relationship over $T$ iterations.
*   **High-Dimensional Data (Kernel Regression):** Extends robust regression to image data (Dogs vs. Frogs) by utilizing a Polynomial Kernel in the dual space, successfully identifying mislabeled or anomalous images as outliers.

### 2. Bayesian Gaussian Mixture Models (GMMs)
*   **Gibbs Sampling Inference:** Implements a fully Bayesian GMM using Gibbs sampling. It samples cluster assignments for each data point and updates the mixture probabilities, means, and covariances based on their respective posterior distributions.
*   **Dynamic Cluster Discovery:** Uses a Dirichlet prior (with parameter $\alpha$) over the mixture probabilities. Even if initialized with a large number of components (e.g., $K=50$), the model naturally prunes empty clusters, leaving only the "active" clusters with significant probability mass.
*   **Flexible Covariance Modeling:** 
    *   *Learned Covariance:* Updates cluster covariances using an MMSE estimation inspired by an Inverse-Wishart prior, ideal for 2D or low-dimensional data.
    *   *Fixed Covariance:* Supports fixed isotropic covariances for high-dimensional data (like images) to ensure computational tractability.
*   **Clustering Evaluation:** Implements the **Cluster Purity** metric to evaluate how well the unsupervised GMM clusters labeled image datasets (e.g., MNIST).

## 📁 Repository Structure

*   **`ex6.py`**: The main execution script. It contains the implementations of the `outlier_regression` function and the `BayesianGMM` class. It also includes the `__main__` block that runs the experiments and generates the visualizations.
*   **`ex6_utils.py`**: A utility module containing:
    *   `BayesianLinearRegression`: A robust class for Bayesian linear regression, supporting both standard basis functions and kernel functions.
    *   Data generation/loading functions (`outlier_data`, `gmm_data`, `load_MNIST`, `load_dogs_vs_frogs`).
    *   Evaluation metrics (`cluster_purity`).
    *   Plotting helpers (`plot_2D_gmm`, `plot_ims`, `get_ellipse`).

## 🚀 Getting Started

### Prerequisites
The project relies on standard scientific computing libraries in Python:
*   `numpy`
*   `scipy`
*   `matplotlib`

You can install them via pip:
```bash
pip install numpy scipy matplotlib
```

### Running the Code
To run the experiments and view the visualizations, simply execute the main script. Make sure the dataset files (e.g., `mnist_partial.npy`, `dogs.npy`, `frogs.npy`) are in the same directory if you intend to run the image clustering/regression parts.

```bash
python ex6.py
```

### Expected Outputs
Upon running the script, you will see a series of plots demonstrating:
1.  **Robust Regression (1D):** A scatter plot showing the convergence of the regression line across $T \in \{0, 1, 5, 10, 25\}$ iterations, successfully ignoring the outliers.
2.  **Image Outliers:** A mosaic plot of the images identified as outliers in the Dogs vs. Frogs kernel regression task.
3.  **Bayesian GMM (2D):** For 5 consecutive runs, you will see:
    *   A bar chart showing the mixture probabilities ($\pi_k$), demonstrating that only a few clusters remain active out of the initial 50.
    *   A 2D scatter plot showing the data points, their assigned clusters, and the learned Gaussian ellipses.

*(Note: The MNIST clustering evaluation in `ex6.py` is commented out by default to save execution time, but can be easily uncommented to test high-dimensional fixed-covariance GMM clustering).*
