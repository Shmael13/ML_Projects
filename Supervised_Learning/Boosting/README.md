# Directory Description
In this directory, we outline implementations and evaluations of boosted decision trees to perform classification on the `diabetes.csv` dataset.

## Boosted Decision Trees Description
Boosting is an ensemble technique that combines multiple weak learners—often decision stumps or shallow decision trees—to create a strong predictive model. In boosting, trees are built sequentially, with each new tree focusing on correcting the mistakes of the previous ones. AdaBoost, one of the most popular boosting algorithms, adjusts the weights of training samples so that misclassified points are given more importance in subsequent trees.

This iterative process reduces bias and variance, enabling boosted trees to achieve higher accuracy than single trees or even bagged trees in many cases. However, boosting can be sensitive to noise and overfitting if not properly tuned.

---

## Advantages
- __Improved Accuracy:__ By sequentially correcting errors, boosting often achieves better performance than a single decision tree or random forest on structured data.
- __Flexibility:__ Boosted trees can model complex relationships by focusing on difficult-to-classify samples.
- __Feature Importance:__ Like decision trees, boosting methods provide feature importance scores, offering insight into which variables most influence predictions.

---

## Disadvantages
- __Sensitive to Noise:__ Because boosting aggressively focuses on misclassified samples, it can overfit to noisy data if regularization parameters like `n_estimators` and `learning_rate` are not properly set.
- __Computational Cost:__ Boosting trains trees sequentially, which can take longer than parallelizable methods like random forests.

# Diabetes Classification with Boosted Decision Trees
## Implementation
This project demonstrates supervised learning using the Pima Indians Diabetes Dataset. The goal is to predict whether a patient has diabetes (`Outcome=1`) or not (`Outcome=0`) using health-related features. An AdaBoost classifier with shallow decision trees as base learners is trained on the data, and performance is evaluated with a confusion matrix and F1 score.

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

1. **Data Preprocessing**
   - The data is read from the CSV file.

   - All features except `Outcome` are standardized using z-score normalization:

     $$z = \frac{x - \mu}{\sigma}$$

     where:
     - $x$ is a feature value,
     - $\mu$ is the mean,
     - $\sigma$ is the standard deviation.

   - The dataset is split randomly, into training and test sets.

2. **Model Initialization and Training**
   - A shallow decision tree (decision stump with `max_depth=1`) is set as the base estimator.
   - An AdaBoost classifier is trained with `n_estimators=100` and `learning_rate=0.5`.
   - A 3D plot shows how F1 score changes across combinations of `n_estimators` and `learning_rate`.

3. **Model Evaluation**
   - A confusion matrix of the AdaBoost classifier is shown, and its F1 score is calculated.
   - An explanation is provided for why the F1 score converges as the number of estimators increases or learning rate changes.

---

## Mathematical Concepts

### 1. Boosting Weight Update
AdaBoost updates sample weights after each weak learner. Misclassified samples receive increased weights to emphasize difficult cases. The weight update for sample $i$ after round $m$ is given by:
$$w_i^{(m+1)} = w_i^{(m)} \cdot e^{\alpha_m \cdot I(y_i \ne h_m(x_i))}$$
where:
- $w_i^{(m)}$ is the weight before the update,
- $\alpha_m$ is the weight of the weak learner,
- $I$ is the indicator function (1 if prediction is incorrect),
- $h_m(x_i)$ is the prediction of the $m$-th weak learner.

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
$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
where:
- Precision $\left(\frac{TP}{{TP} + {FP}}\right)$
- Recall $\left(\frac{TP}{{TP} + {FN}}\right)$

A **weighted average F1 score** accounts for class imbalance by weighting each class’s F1 by its support (number of true instances).
