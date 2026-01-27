This repository contains end-to-end implementations of selected ML and DL models written from scratch using only NumPy and basic Python utilities.
The goal of this project is not performance, but deep understanding. Each model is implemented with explicit forward and backward passes, custom loss functions, and manually derived gradients.

Emphasis is placed on:

1. Mathematical formulation
2. Optimization behavior
3. Training stability
4. Failure modes and limitations


## Implemented Models : 

### 1. Multilayer Perceptron (MLP)
- Feed Forward neural network
- ReLU fn
- Softmax output layer
- Cross-entropy loss
- Full backpropagation implemented manually

### 2. Softmax Classifier (Multiclass Classification)
- Linear classifier with softmax normalization
- Cross-entropy loss
- Gradient descent

### 3. Principal Component Analysis (PCA)
- Covariance matrix computation
- Eigenvalue–eigenvector decomposition
- Dimensionality reduction via projection


## Mathematical Derivations

All models are accompanied by handwritten mathematical derivations, including:
- Loss functions
- Gradient calculations
- Backpropagation equations
- Optimization updates

These derivations are available in: 
Proofs.pdf
