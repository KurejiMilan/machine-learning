# knn algorithm for classification

import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny',
           'Overcast', 'Overcast', 'Rainy']

temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']

play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

# need to perform label encoding for that we are making a label encoder object

le = preprocessing.LabelEncoder()
encodedWeather = le.fit_transform(weather)
encodedTemp = le.fit_transform(temp)
print(encodedWeather)
print(encodedTemp)

# combine the weather and temp into a single tuple

combinedFeature = tuple(zip(encodedWeather, encodedTemp))
print(combinedFeature)

# building Knn classifier, by creating a KNN classifier object
knnClassifierModel = KNeighborsClassifier(n_neighbors=3)
knnClassifierModel.fit(combinedFeature, play)

predicted = knnClassifierModel.predict([[0, 2]])
print(predicted)
