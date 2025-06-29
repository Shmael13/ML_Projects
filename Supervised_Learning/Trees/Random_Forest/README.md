# Directory Description
In this directory, we outline implementations and evaluations of random forests to perform classification on the `diabetes.csv` dataset.

## Random Forest Description
A random forest is an ensemble learning algorithm that builds a collection of decision trees during training. For classification, it aggregates their predictions through majority voting; for regression, it averages their outputs. By combining many weak learners (decision trees), random forests reduce variance and improve generalization compared to a single tree.

Random forests train each tree on a bootstrap sample of the training data, introducing randomness. At each split in a tree, a random subset of features is considered, which further decorrelates the trees. This randomness prevents overfitting and makes the forest robust to noise in the data.

---

## Advantages 
- __Robustness:__ By averaging predictions from many uncorrelated trees, random forests reduce overfitting and yield better performance on unseen data compared to a single decision tree.
- __Feature Importance:__ Random forests naturally provide estimates of feature importance, helping identify which variables contribute most to predictions.
- __Flexible:__ They can be used for both classification and regression and handle large datasets with high-dimensional feature spaces.

---

## Disadvantages
- __Reduced Interpretability:__ Unlike a single decision tree, random forests act as black-box models. It is challenging to visualize the decision process across hundreds of trees.
- __Computationally Intensive:__ Training and predicting with a large number of trees can be slower and require more memory compared to simpler models.

# Diabetes Classification with Random Forests
## Implementation
This project demonstrates supervised learning using the Pima Indians Diabetes Dataset. The goal is to predict whether a patient has diabetes (`Outcome=1`) or not (`Outcome=0`) using health-related features. A random forest classifier is trained on the data, and performance is evaluated with a confusion matrix and F1 score.

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

   - The dataset is split randomly into a training and test set.

2. **Model Initialization and Training**
   - A random forest classifier with 100 trees and a maximum depth of 5 is trained on the scaled training data.
   - Feature importance scores are calculated to show which variables were most influential in predictions.

3. **Model Evaluation**
   - A confusion matrix of the random forest is shown, and its F1 score is calculated.
   - An explanation is given for how increasing the number of trees stabilizes the F1 score as predictions average over more learners.

---

## Mathematical Formulas

### 1. Random Forest Voting
For classification, each tree predicts a class label. The forest’s output is the class with the highest number of votes among all trees:
$$\hat{y} = mode( \{\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_n\} )$$
where $\hat{y}_i$ is the prediction of tree $i$.

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

---

## Results

- The confusion matrix and feature importance chart provide insight into model performance and highlight which features contributed most to diabetes prediction.

---
