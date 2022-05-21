import numpy as np
from collections import counter

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-xn)**2))

class knn:
    def __init__(self,k=3):
        self.k = k

    def fit(self,x,y):
        self.xTrain = x
        self.yTrain = y

    def predict(self,x):
        y_pred = [self._predict(x) for x0 in x]
        return np.array(y_pred)

    def _predict(self,x):
        distances = [euclidean_distance(x,xTrain) for xTrain in self.xTrain]
        k_idx = np.argsort(distances)[:self.k]