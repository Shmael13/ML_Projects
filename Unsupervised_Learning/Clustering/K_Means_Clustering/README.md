# K-Means Clustering from Scratch on the Iris Dataset

This project implements a **custom K-Means clustering algorithm** and applies it to the Iris dataset. It clusters the samples based on **petal length** and **petal width**, visualizes how the centroids move during training, and compares the predicted clusters to the true species labels.

---

## `KMeansClustering` Algorithm Description

This implementation builds the K-Means algorithm manually with the following algorithm:

1. **Initialization**  
   - Randomly selects `n_clusters` points from the data as initial centroids.

2. **Assignment Step**  
   - For every point, calculates the Euclidean distance to each centroid.
   - Assigns each point to the cluster of its nearest centroid.

3. **Update Step**  
   - Updates each centroid to be the mean of the points assigned to its cluster.

4. **Convergence**  
   - Computes the total shift (Euclidean distance) of centroids between iterations.
   - Stops if the maximum number of iterations is reached or the centroid shift is below the tolerance threshold `tol`.

5. **Centroid Tracking**  
   - Stores the centroids at every iteration so their movement over time can be visualized.

---

## Visualizations

The script produces two plots:
1. **K-Means Clustering on Iris**: shows data points colored by predicted cluster, with lines tracing the movement of centroids during iterations and final centroid positions marked.
2. **Expected Clusters from Model**: shows true species clusters from the Iris dataset, colored by actual labels.

By comparing the actual clusters with the clusters created by the K-Means Algorithm, we can evaluate how well our model performs/

---
