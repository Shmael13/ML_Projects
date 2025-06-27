# Diabetes Classification with Decision Trees
## Description

A decision tree is a non-parametric supervised learning algorithm. It is used in both classification and regression. It consists a tree structure, with a single root node, with many possible internal and leaf nodes. Each node can have 0, 1, or 2 children nodes.

The purpose of each node is to split the dataset into the smallest 'pure' subset it can, by splitting according to the features. The outgoing branches from each node 'feed' into its child nodes. The leaf nodes encapsulate all the possible points within the dataset. However, with more complex datasets, splitting into 'pure' data is not easy.

In order to find the optimal split points within the tree, a divide and conquer algorithm is used, which performs a greedy search for the points. The process of splitting is repeated recursively until all the features have been classified under certain labels. Depending on the complexity of the dataset, all the points may or may not be classified into pure leaf nodes. As a tree grows larger, the leaf nodes contain fewer and fewer points. This data fragmentation can lead to overfitting. 
To overcome this, sometimes pruning is applied - whereby branches splitting on features with low importance are removed. Another way to imporve accuracy is using ensemble methods like Boosting. 

### Advantages 
- Explainable Behavior: The trees split data according to specific and easy-to-understand logic on the features. This, alongside the visual representations of the trees, help making understanding the reason for their behavior easier. It also helps understand which features are given most importance by the model.
- Flexible: Decision  can be leveraged for both classification and regression tasks. Moreover, it isn't sensitive to corelated variables. For any two highly corelated variables, it will only choose one to split on.
 
### Disadvantages
- Overfitting: Complex decision trees often to overfit the data, and don't generalize well to newer data. This can be avoided through the processes of pruning, which halts growth or removes nodes at different stages when there isn't enough data within a node or subtree to justify splitting.
- Computationally Expensive: Because decision trees use greedy search during construction, they can be a lot more expensive to train than other algorithms.

## Implementation
This project demonstrates supervised learning using the Pima Indians Diabetes Dataset. The goal is to predict whether a patient has diabetes (`Outcome=1`) or not (`Outcome=0`) using health-related features. A decision tree classifier is trained on the data, and performance is evaluated with a confusion matrix and F1 score.

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
   - The Data is read from the csv file

   - All features except `Outcome` are standardized using the z-score normalization:

     $$z = \frac{x - \mu}{\sigma}$$

     where:
     - $x$ is a feature value,
     - $\mu$ is the mean,
     - $\sigma$ is the standard deviation.
   
   - The dataset is split randomly:
     - 75% for training,
     - 25% for testing.

3. **Model Initialization and Training**
   - A decision tree of maximum depth 3 is trained on the scaled training data.
   - The trained tree is plotted, showing splits on features and class distributions at the leaves.
4. **Model Evaluation**
   - A confusion matrix of the tree is shown, and its F1 score is calculated
   - An explanation is given for why F1 scores converge when we continue increasing the max depth.

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

- The F1 score on the test set was **0.601**.
- Confusion matrix and decision tree visualization provide insight into model behavior.

---
