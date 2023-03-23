---
title: "Workshop Planning"
author: "Musashi Jacobs-Harukawa"
---


# Rough Planning

1. Motivation
    - Who is this lecture for? (Me a few years ago)
    - What is DL?
        - Approach to using computers to learn patterns in data (machine learning)
        - based on a modular set of tools loosely inspired by biological neurons.
        - Basis of many advanced systems seen today. (Instinct to call it "black box")
    - Why should socsci care about DL?
        - Utility for modelling
        - Effect on society
    - Why look at the fundamentals?
        - You can (increasingly) get by with minimal understanding how it works
        - But we are interested in characterizing "full" behavior of methods
    - Learning Objectives:
        - The basic building block of neural networks: neurons.
        - How to learn from data with optimization.
        - How and why ensembling neurons is powerful.
    - Format: Lecture + Supplementary Workbook for Implementation
        - Trialled doing both side-by-side, but too many new things at once.
        - You don't need me for the workbook, but I recommend that you use it and feel free to contact me with questions.
    - Next Time:
        - Scaling up further: modular architectures and mechanisms.
        - Generating sequences: autoregressive language generation.
        - Architectures for sequences: RNNs and Transformers.
    - tl;dr (One-Slide Version)
        - _Neural networks are basically nested regression models with discontinuities_.
        - _Backpropagation is a technique that allows us to calculate the loss gradient of every parameter in a network_.
        - _Loss gradients link each parameter to the overall accuracy of the model_.
        - _Stochastic gradient descent is an algorithm for updating parameters using loss gradients to improve accuracy_.
        - _By combining lots of regressions and introducing regressions, we can learn increasingly complex patterns in data_.
2. Simple Case: One X, One Y
    - _Neural networks are basically nested regression models with discontinuities_.
    - Begin with some sample data. We can fit a linear model to measure the average association between the variables.
    - Let's start with a random line. How do we make the line fit to the data?



# Rough Notes

## Format

- Lecture focuses on conceptual aspects with animated explanations
- Coding exercise offered as a take-away module that can be used for self-study.

## Objective

- 




# General Notes

Broad headers/concepts/ways to organize:

- Disclaimers/Expectations:
	- Why Python/PyTorch? It's standard. Plus I find pytorch keeps the theory and implementation pleasantly close.


- Training Loop: How to Fit the Model to the Data
	- Forward Pass (Prediction)
	- Loss Calculation
	- Backprop
	- Parameter update

- Modules and Composability: How to Build More Flexible Models
	- Modules are blueprints
	- Training modules with batches, optimizers and dataloaders
 

 # Graveyard

\begin{align}
\frac{\delta L}{\delta w_1} &= \frac{\delta L}{\delta\hat{y}} \frac{\delta \hat{y}}{\delta w_2}
							&= 2(y-\hat{y}) h_1

\end{align}

- $\frac{\delta L}{\delta w_2} = \frac{\delta L}{\delta\hat{y}} \frac{\delta \hat{y}}{\delta w_2}$
- $\frac{\delta L}{\delta w_1} = \frac{\delta L}{\delta\hat{y}} \frac{\delta\hat{y}}{\delta h_1} \frac{\delta h_1}{\delta w_1}$


- $L = (y - w_{2}(w_{1}X + b_{1}) + b_{2})^2$
- $h_1 = w_{1}X + b_{1}$
- $\frac{\delta h_1}{\delta w_1} = X$
- $\frac{\delta L}{\delta h_1} = -w_2$
- $\frac{\delta L}{\delta w_1} = \frac{\delta L}{\delta h_1} \frac{\delta h_1}{\delta w_2}$
