#Knn with own dataset
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

weather =['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy']

temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

le = preprocessing.LabelEncoder()

EncodedWeather = le.fit_transform(weather)
EncodedTemp = le.fit_transform(temp)
label = le.fit_transform(play)

features = list(zip(EncodedWeather,EncodedTemp))

model = KNeighborsClassifier(n_neighbors=3)

model.fit(features,label)

predicted = model.predict([[0,2]])

print(predicted)