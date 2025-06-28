# Directory Description
In this directory, we outline implementations and evaluations of decision trees to perfrom regression and classification tasks, on the `Advertising.csv` and `diabetes.csv` datasets respectively. 

## Decision Tree Description
A decision tree is a non-parametric supervised learning algorithm. It is used in both classification and regression. It consists a tree structure, with a single root node, with many possible internal and leaf nodes. Each node can have 0, 1, or 2 children nodes.

The purpose of each node is to split the dataset into the smallest 'pure' subset it can, by splitting according to the features. The outgoing branches from each node 'feed' into its child nodes. The leaf nodes encapsulate all the possible points within the dataset. However, with more complex datasets, splitting into 'pure' data is not easy.

In order to find the optimal split points within the tree, a divide and conquer algorithm is used, which performs a greedy search for the points. The process of splitting is repeated recursively until all the features have been classified under certain labels. Depending on the complexity of the dataset, all the points may or may not be classified into pure leaf nodes. As a tree grows larger, the leaf nodes contain fewer and fewer points. This data fragmentation can lead to overfitting. 
To overcome this, sometimes pruning is applied - whereby branches splitting on features with low importance are removed. Another way to imporve accuracy is using ensemble methods like Boosting. 

---

## Advantages 
- __Explainable Behavior:__ The trees split data according to specific and easy-to-understand logic on the features. This, alongside the visual representations of the trees, help making understanding the reason for their behavior easier. It also helps understand which features are given most importance by the model.
- __Flexible:__ Decision can be leveraged for both classification and regression tasks. Moreover, it isn't sensitive to corelated variables. For any two highly corelated variables, it will only choose one to split on.

---
 
## Disadvantages
- __Overfitting:__ Complex decision trees often to overfit the data, and don't generalize well to newer data. This can be avoided through the processes of pruning, which halts growth or removes nodes at different stages when there isn't enough data within a node or subtree to justify splitting.
- __Computationally Expensive:__ Because decision trees use greedy search during construction, they can be a lot more expensive to train than other algorithms.

