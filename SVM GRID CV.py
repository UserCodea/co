import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

bankdata = pd.read_csv("bill_authentication.csv")
bankdata.shape
bankdata.head()
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


grid = {
    'C':[0.01,0.1,1,10],
    'kernel' : ["linear","poly","rbf","sigmoid"],
    'degree' : [1,3,5,7],
    'gamma' : [0.01,1]
}
svm  = SVC ()
gridcv = GridSearchCV(SVC(), grid)




gridcv.fit(X_train,y_train)

# print best parameter after tuning
print(gridcv.best_params_)

# print how our model looks after hyper-parameter tuning
print(gridcv.best_estimator_)
