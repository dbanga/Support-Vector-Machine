# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:28:16 2018

@author: dishant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importdataset
dataset=pd.read_csv('Social_Network_Ads.csv')

#changing categorical varo=ible
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
dataset.Gender=labelencoder.fit_transform(dataset.Gender)

#splitiing into dependenat and independant varibale

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 4].values

#Splitting into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.25, random_state=0)


#scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#Classifier
from sklearn.svm import SVC
classifier=SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, Y_train )

#predicting 
Y_pred = classifier.predict(X_test)

#cONFUSIONMatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, Y_pred)
print(cm)

