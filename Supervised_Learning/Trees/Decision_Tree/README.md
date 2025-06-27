# Diabetes Classification with Decision Trees

This project demonstrates a supervised learning pipeline using the Pima Indians Diabetes Dataset. The goal is to predict whether a patient has diabetes (`Outcome=1`) or not (`Outcome=0`) using health-related features. A decision tree classifier is trained on the data, and performance is evaluated with a confusion matrix and the weighted F1 score.

## Dataset

The dataset `diabetes.csv` includes these features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target variable)

---

## Workflow

1. **Data Scaling**
   - All features except `Outcome` are standardized using the z-score normalization:
     $$
     z = \frac{x - \mu}{\sigma}
     $$
     where:
     - $x$ is a feature value,
     - $\mu$ is the mean,
     - $\sigma$ is the standard deviation.

2. **Train/Test Split**
   - The dataset is split randomly:
     - 75% for training,
     - 25% for testing.

3. **Decision Tree Classifier**
   - A decision tree of maximum depth 3 is trained on the scaled training data.

4. **Visualization**
   - The trained tree is plotted, showing splits on features and class distributions at the leaves.

5. **Evaluation**
   - Predictions on the test set are compared with true labels using a confusion matrix and the weighted F1 score.

---

## Mathematical Formulas

### 1. Decision Tree Splitting Criterion (Gini Impurity)
To decide the best split at each node, the decision tree minimizes the **Gini impurity**, defined for a node $t$ as:
$Gini(t) = 1 - \sum_{k=1}^{K} p_{k}^{2}$
where:
- $K$ is the number of classes,
- $p_k$ is the proportion of samples of class $k$ in node $t$.

A perfect node (pure class) has $Gini=0$, and maximum impurity (even class distribution) has $Gini=0.5$ for two classes.

---

### 2. Confusion Matrix
A confusion matrix summarizes predictions vs. actual labels:
|            | Predicted Positive | Predicted Negative |
|------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP)     | False Negative (FN)   |
| **Actual Negative** | False Positive (FP)    | True Negative (TN)    |

---

### 3. F1 Score
The F1 score balances precision and recall:
$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
where:
- Precision $(\frac{TP}{{TP} + {FP}})$
- Recall $(\frac{TP}{{TP} + {FN}})$

A **weighted average F1 score** accounts for class imbalance by weighting each class’s F1 by its support (number of true instances).

---

## Results

- The weighted average F1 score on the test set was **0.704**.

- Confusion matrix and decision tree visualization provide insight into model behavior.

---
