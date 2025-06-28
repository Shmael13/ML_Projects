# Linear and Logistic Regression

This directory contains implementations of **Linear Regression** and **Logistic Regression** from scratch. These are implemented on the *Advertising.csv* dataset, to predict the amount of Sales based on spending on **TV**, **radio**, and **newspaper** advertising.

## Table of contents:
- [Single Neuron Class](#single-neuron-class)
- [Model Evaluation](#model-evaluation)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Residual Analysis](#residual-analysis)

---

## Single Neuron Class
The `SingleNeuron` class fits data using gradient descent. It adjusts its weights iteratively to minimize the error between predicted and actual values.

---

### Mathematical Foundations

#### Net Input Calculation

For an input vector **X**, weights **w**, and bias **b**, the neuron computes the net input as:
$z = X \cdot w + b$
This represents the linear combination of inputs and weights.

#### Activation Function

The activation function transforms the net input into the final output.

#### Error Calculation

The error for each prediction is defined as:
$\epsilon = \hat{y} - y$

#### Loss Function (Per Epoch)

The mean squared error (MSE) over the training set is:
$Loss= \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$

#### Gradient Descent Weight Updates

Gradient descent updates each weight and the bias in the direction that reduces the error:
$w_j \leftarrow w_j - \alpha \cdot \epsilon \cdot x_j$
$b \leftarrow b - \alpha \cdot \epsilon$
where:
- $\alpha$ is the learning rate.
- $x_j$ is the j-th feature value.

---

## Model Evaluation

### Error Metrics
Evaluate model performance with standard regression metrics:

- **Mean Squared Error (MSE)**  
  $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

- **Root Mean Squared Error (RMSE)**  
  $RMSE = \sqrt{MSE}$

- **Mean Absolute Error (MAE)**  
  $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

- **R-squared (R²)**  
  $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$

Below are some scored metrics, which show how our model is performing when tested

- Mean Squared Error (MSE): 
    - Average squared error between expected (y_test) and predicted (y_pred) outputs
    - Smaller MSE means a better fit
    - Errors are squared, so it penalizes large errors heavily. This makes it sensitive to outliers
    - Lower is better
- Root Mean Squared Error (RMSE):
    - Square root of MSE
    - Has the same units as the target variable (sales). This makes it easier to interpret
    - RMSE approximates average magnitude of prediction error
    - Lower is better
- Mean Absolute Error (MAE):
    - Average absolute difference between expected (y_test) and predicted (y_pred) outputs
    - Less sensitive to outliers than RMSE
    - Lower is better
- R Squared / Coefficient of Determinanation (R2):
    - Measure of how well model fits data
    - 1 indicates perfect fit of model to data
    - 0 indicates model doesn't explain any veriability in data
    - Higher is better

### Residual Analysis
While the error metrics used above may tell us how large our errors are on average, they don't display how the errors are distributed. Residual plots display this, and help us figure out whether errors are randomly scattered around zero, or if they show systemic patterns.
Residuals are calculated as:
$Residual = y_{test} - y_{pred}$

**Interpretation:**
- Positive residual → prediction too low.
- Negative residual → prediction too high.
- Near-zero residual → accurate prediction.

![Ideal_Residuals](ideal_residual.png)

An ideal residual plot:
- Residuals scatter symmetrically around zero.
- No clear patterns.
- Residuals remain close to 0.

![Poor_Residual](poor_residual.png)

If residuals _do_ show patterns, they may display information we can use to correct our model. Some examples certain residual plots and some corrections we might apply to them are:
- A curve or trend $\rightarrow$ model misses nonlinearity in data
- Points mostly positive or negative in some ranges $\rightarrow$ model systemically over- or under-predicts
- Large magnitude in a few points $\rightarrow$ outliers in data
- Funnel shape $\rightarrow$ may indicate missing variable
