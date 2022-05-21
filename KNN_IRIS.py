from sklearn.neighbors import KNeighborsClassifier
from  sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris= load_iris()

x = iris.data
y = iris.target

xTrain ,  xTest , yTrain , yTest = train_test_split(x,y,test_size=0.2,random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(xTrain,yTrain)

predict = knn.predict(xTest)

print(predict)