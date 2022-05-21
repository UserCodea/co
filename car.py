import numpy as np
import pandas as pd
from sklearn.naive_bayes import  CategoricalNB
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set()
car_df = pd.read_csv("car.data", names=['buying','maint','doors','persons','lug_boot','safety','class'])
print(car_df.head())
print('=' * 30)
print(car_df.info())
print('=' * 30)
print(car_df.dtypes)
print('=' * 30)
print(car_df.isnull().sum())
print('=' * 30)
print( car_df.shape[0])
print('=' * 30)

def count_plot(df, columns):
    plt.figure(figsize=(15, 10))
    for indx, var  in enumerate(columns):# adds a counter returns an enumerate object (0,buying) (1,maint) (2,doors)
        plt.subplot(2, 3, indx+1)# number of rows ,number of coulumns,index of current plot
        g = sns.countplot(df[var], hue= df['class'])# show the counts of observations in each feature
    plt.tight_layout()
    plt.show()

features = car_df.columns.tolist()
features.remove('class')
print(features)   
count_plot(car_df, features)
for var in features:
    print(car_df[var].value_counts().sort_values(ascending=False) / car_df.shape[0])
    print('=' * 30)
    print()

encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(car_df[features])
car_df_encoded = pd.DataFrame(data_encoded, columns=features)
print(car_df_encoded .head())

encoder = LabelEncoder()
target_encoded = encoder.fit_transform(car_df['class'])
car_df_encoded['class'] = target_encoded
print(car_df_encoded['class'].head())
X_train, X_test, y_train, y_test = train_test_split(car_df_encoded.drop('class', axis=1), car_df_encoded['class'], test_size=0.3)
cnb = CategoricalNB()
cnb.fit(X_train, y_train)
