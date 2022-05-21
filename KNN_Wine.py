from sklearn.neighbors import KNeighborsClassifier
from  sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from  sklearn import metrics

win = load_wine()
xTrain , xTest , yTrain , yTest = train_test_split(win.data,win.target,test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(xTrain,yTrain)
predict = knn.predict(xTest)

print("Accuracyy:",metrics.accuracy_score(yTest,predict))