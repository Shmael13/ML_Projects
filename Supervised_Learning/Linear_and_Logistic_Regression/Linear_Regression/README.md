# Linear Regression on Advertising Dataset

This Jupyter Notebook demonstrates how **Linear Regression** can be used for regression on an Advertising dataset, predicting sales based on spending on **TV**, **radio**, and **newspaper** advertising.

## Table of contents:
- [Workflow](#workflow)
- [Data Preprocessing](#data-preprocessing)
- [Model Implementation](#model-implementation)
  - [Single Neuron Class](#single-neuron-class)
  - [Training Model with Linear Activation](#training-model-with-linear-activation)
- [Model Evaluation](#model-evaluation)
  - [Error Metrics](#error-metrics)
  - [Actual VS Predicted Sales](#actual-vs-predicted-sales)
  - [Residual Analysis](#residual-analysis)
- [Visualization](#visualization)
- [Conclusion](#conclusion)
---

## Workflow

The notebook follows three phases:
1. **Data Preprocessing**
2. **Model Implementation**
3. **Model Evaluation**

---

## Data Preprocessing

This phase prepares the dataset by:
- Importing essential libraries (`mpl_toolkits.mplot3d`, `sklearn`, `matplotlib`, `pandas`, `numpy`).
- Reading `Advertising.csv`.
- Scaling numeric features (`TV`, `radio`, `newspaper`) with `MinMaxScaler`.
- Splitting data into training and test sets using `train_test_split`.

---

## Model Implementation
This section details the implementation of a **single neuron model** performing linear regression through gradient descent.

### Single Neuron Class
The Single Neuron Class models a linear regression. A description of the class is given in the parent directory's *README.md* file.

### Training Model with Linear Activation
For linear regression, we use the identity function:
$\hat{y} = z$

Since we know that the formula for linear regression is:
$\hat{y} = \mathbf{X} \cdot \mathbf{w} + b$

Given what we know about the implementation of the Single Neuron, this effectively means that when we use the linear activation function, we get:

$\boxed{\hat{y} = z = X \cdot w + b}$

---

## Model Evaluation

### Error Metrics
Evaluate model performance with standard regression metrics:

- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  
- **Mean Absolute Error (MAE)**  
- **R-squared (R²)**  

---

### Actual VS Predicted Sales
Ideally, we want a perfect mapping between actual and predicted data. Some things to take note of are:
- Systemic Deviation from line $\rightarrow$ biased model
- High scatter $\rightarrow$ poor predictive power

---

### Residual Analysis
A description of what Residual Analysis is, is provided in the parent directory. The specific outputs of the residual in the case of our Linear Regression model are expressed within the notebook.

## Conclusion
The notebook shows:
- A from-scratch implementation of a Linear Regressor.
- An evaluation of the performance of our regressor on a realistic dataset.

---

## References
Images come from: https://www.qualtrics.com/support/stats-iq/analyses/regression-guides/interpreting-residual-plots-improve-regression/
