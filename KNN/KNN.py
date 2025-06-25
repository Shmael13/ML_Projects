import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# If you have a bunch of stocks w movement, stocks are corellated - nearest neighbors can find which stocks are most similar/dependant on each other based on country of origin, region, sector, time, etc. From that, you can say a given stock has similarity to others near it.
# Very obvious 
# Ok to have scripts, but give explanations in notebook.
# https://medium.com/@thornexdaniel/ml-algorithms-in-the-markets-ddbff48c7e0

data = pd.read_csv("wisconsin_breast_cancer.data")
print(data.head())

y = np.array(data.iloc[: , 1])
y = y.reshape(-1,1)
y = np.where(y == 'M', 0, 1)

X = np.array(data.iloc[: , 2:4])
print(f"Example X value: {X[0]}")
print(f"Inputs matric shape: {X.shape}")

#Ask Prof Davila why changing the testsize doesn't significantly change the errors created. Is my code wrong?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

print(f"Example y_trian value:  {y_train[0]}")
print(f"Label X_train shape: {X_train.shape}")

class KNN(object):
    def __init__(self, distance_function):
        self.dist_f = distance_function
        
    def get_neighbors(self, k, point, train_features, train_labels):
        neighbors = []
        for p, label in zip(train_features, train_labels):
            d = self.dist_f(p, point)
            #Do not add point to the list of neighbors
            temp_data = [p, label, d]
            neighbors.append(temp_data)

        #Sort the neighbors based on the distance
        neighbors.sort(key=lambda x: x[-1])
        return neighbors[:k]

    def predict(self, point, train_features, train_labels, k):
        neighbors = self.get_neighbors(k, point, train_features, train_labels)
        labels = [x[1] for x in neighbors]
        return max(labels, key=labels.count)
                    

def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

def classification_error(test_features, train_features, test_labels, train_labels, model, k):
    error = 0
    for point, label in zip(test_features, test_labels):
        error += (label.item() != model.predict(point, train_features, train_labels, k))

    return error/len(test_features)


knn = KNN(manhattan_distance)
y_pred = knn.predict(X_train[0], X_train, y_train, 3)

errors = classification_error(X_test, X_train, y_test, y_train, knn, 3)

print(f"y_pred: {y_pred}")
print(f"Errors: {errors}")

def plot_decision_boundary(model, X_train, y_train, k, h=0.1):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.array([
        model.predict(np.array([x, y]), X_train, y_train, k)
        for x, y in zip(xx.ravel(), yy.ravel())
    ])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), edgecolors='k')
    plt.title(f"KNN Decision Boundary (k={k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(knn, X_train, y_train, k=3)

