#SVM

#libraries import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten

#data set import
dataset=pd.read_csv('Social_Network_Ads.csv')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
dataset.Gender=labelencoder.fit_transform(dataset.Gender)

#onehotencoder=OneHotEncoder(categorical_features=dataset.Gender)
#dataset=onehotencoder.fit_transform(dataset.Gender).toarray()


#dataset.Gender=dataset.Gender.astype("category").cat.codes


X = dataset.iloc[:, :3].values
Y = dataset.iloc[:, 4].values


#splitting dataset into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


##SVM classfier
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train, Y_train)

#predicting the test (validation)
Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
CM=confusion_matrix(Y_test, Y_pred)
print(CM)