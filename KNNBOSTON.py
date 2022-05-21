from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

BostonData = load_boston()
data = BostonData.data
labels = BostonData.target

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle =True, random_state=2021)

KNeighborsRegressorModel = KNeighborsRegressor(n_neighbors = 8, weights='uniform',algorithm = 'auto')
KNeighborsRegressorModel.fit(X_train, y_train)
print('Train Score is : ' , KNeighborsRegressorModel.score(X_train, y_train))
print('Test Score is : ' , KNeighborsRegressorModel.score(X_test, y_test))