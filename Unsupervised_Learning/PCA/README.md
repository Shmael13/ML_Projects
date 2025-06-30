# PCA on the Penguins Dataset

This project applies **Principal Component Analysis (PCA)** to the Palmer Penguins dataset. It includes custom preprocessing (standardizing features and encoding categories) and visualization of the principal components, helping illustrate how PCA can reveal patterns in complex data.

---

## What is PCA?

**Principal Component Analysis (PCA)** is a mathematical technique used to reduce the number of features in a dataset while preserving as much variation (information) as possible. It does this by:
1. Centering and scaling each feature so they’re comparable.
2. Finding new axes (principal components) along which data varies the most.
3. Projecting data onto these new axes so you can use fewer dimensions without losing information.

Mathematically, PCA transforms the data by computing the eigenvectors of the covariance or correlation matrix of the standardized data. These eigenvectors (principal components) capture the directions of maximum variance, ordered by importance.

---

## Dataset

The project uses the [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/), which contains measurements of penguins (bill length, flipper length, body mass) alongside Species and Sex information.

---

## Code Breakdown

- Loads and cleans the data (drops rows with missing values).  
- One-hot encodes `species` and `island`.
- Standardizes selected numerical features (e.g., bill length, flipper length) with `StandardScaler`.  
- Combines numerical and encoded categorical features into a single DataFrame.  
- Applies PCA to the standardized features.
- Visualizes:
   - The first two principal components (those explaining the most variance).
   - Later components (those explaining the least variance) to show how little information they contain.
- Colors the plots by the sex of the penguins for comparison.
