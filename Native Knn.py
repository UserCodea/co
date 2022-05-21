from sklearn.neighbors import KNeighborsClassifier
from  sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def _init_(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[: self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


iris= load_iris()

x = iris.data
y = iris.target

xTrain ,  xTest , yTrain , yTest = train_test_split(x,y,test_size=0.2,random_state=42)

knn = KNN()
knn.k=7
knn.fit(xTrain,yTrain)
predict = knn.predict(xTest)
print(predict)