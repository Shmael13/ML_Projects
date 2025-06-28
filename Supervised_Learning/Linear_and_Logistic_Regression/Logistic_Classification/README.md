# Logistic Classification on Advertising Dataset

This project demonstrates using Logistic Classification on the Pima Indians Diabetes Dataset. The goal is to predict whether a patient has diabetes (`Outcome=1`) or not (`Outcome=0`) using health-related features. A decision tree classifier is trained on the data, and performance is evaluated with a confusion matrix and F1 score.

## Table of contents:
- [Workflow](#workflow)
- [Data Preprocessing](#data-preprocessing)
- [Model Implementation](#model-implementation)
  - [Single Neuron Class](#single-neuron-class)
  - [Training Model with Sigmoid Activation](#training-model-with-sigmoid-activation)
- [Model Evaluation](#model-evaluation)
  - [Confusion Matrix](#confusion-matrix)
  - [F1 Score](#f1-score)
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
- Reading `diabetes.csv`.
- Scaling numeric features with `MinMaxScaler`.
- Splitting data into training and test sets using `train_test_split`.

---

## Model Implementation
This section details the implementation of a **single neuron model** performing logistic classification through gradient descent.

### Single Neuron Class
The Single Neuron Class models a logistic classification. A description of the class is given in the parent directory's *README.md* file.

### Training Model with Logistic Activation
For logistic classification, we use the sigmoid activation function:
$\hat{y}= \frac{1}{1 + e^{-z}}$

Since we know that the formula for logistic classification is:
$$P(Y=1|x) = \sigma(\beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n)$$

Given what we know about the implementation of the Single Neuron, by putting the sigmoid function as the activation, we perform precisely this logistic classification.

---

## Model Evaluation

### Confusion Matrix
A confusion matrix summarizes predictions vs. actual labels:
|            | Predicted Positive | Predicted Negative |
|------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP)     | False Negative (FN)   |
| **Actual Negative** | False Positive (FP)    | True Negative (TN)    |


---

### F1 Score
The F1 score balances precision and recall:
$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
where:
- Precision $(\frac{TP}{{TP} + {FP}})$
- Recall $(\frac{TP}{{TP} + {FN}})$

A **weighted average F1 score** accounts for class imbalance by weighting each class’s F1 by its support (number of true instances).

## Conclusion
The notebook shows:
- A from-scratch implementation of a Logistic Classifier.
- An evaluation of the performance of our classification on a realistic dataset.
