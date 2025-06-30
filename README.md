# Machine Learning Projects
Creator: Ismail Syed

This repository contains a collection of machine learning algorithms. Machine learning can be subdivided into two major categories - supervised and unsupervised. The repository is also divided in this way. It contains a mix of self-implemented modules, and modules implemented using the famous sklearn library.

## Supervised vs Unsupervised
Supervised learning is when an algorithm is trained with pre-determined labels and features. A label is the ideal prediction we want our model to output, and features are the inputs given to the models to obtain this corresponding label.
Unsupervised learning is when an algorithm is used to find underlying structures within data to either reduce the dimensionality (number of features) or find clusters. In this type of learning, no labels are provided.

### Regression vs Classification
Regression finds the best line to minimize some cost function. This line can then be used to predict a continuous label based on any input features. 

Classification uses features to predict which category the datapoint belongs to (within a predetermined set of categories). 

### Clustering vs Dimensionality Reduction
Clustering groups the dataset into a pre-specified number of clusters. This helps us understand how the data is segmented, or which group future data will belong to.Since clustering is unsupervised, the number and types of labels are completely unkown to the model, and the programmer only specifies the number of potential clusters.

Dimensionality reduction reduces the number of features in a dataset while preserving as much important information as possible. This helps us simplify complex data, visualize patterns in 2D or 3D, and speed up training for machine learning models. Like clustering, dimensionality reduction is unsupervised so it has no information about the labels — it uses learns the directions of maximum variance in the data to find the most informative features within the dataset.
