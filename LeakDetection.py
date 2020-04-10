import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

data=pd.read_csv('random dataset.csv')

X=data.drop('Leak',axis=1)
Y=data['Leak']

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y)

logregmodel=LogisticRegression().fit(X_train,Y_train)
lrscore=logregmodel.score(X,Y)
Y_pred = logregmodel.predict(X_test)
lraccuracy=metrics.accuracy_score(Y_test,Y_pred)

knnmodel=KNeighborsClassifier(n_neighbors = 1).fit(X_train,Y_train)
knnscore=knnmodel.score(X,Y)
Y_pred = knnmodel.predict(X_test)
knnaccuracy=metrics.accuracy_score(Y_test,Y_pred)

svmmodel=SVC().fit(X_train,Y_train)
svmscore=svmmodel.score(X,Y)
Y_pred = svmmodel.predict(X_test)
svmaccuracy=metrics.accuracy_score(Y_test,Y_pred)

nbmodel=GaussianNB().fit(X_train,Y_train)
nbscore=nbmodel.score(X,Y)
Y_pred = nbmodel.predict(X_test)
nbaccuracy=metrics.accuracy_score(Y_test,Y_pred)

dtmodel=DecisionTreeClassifier(max_depth=3,random_state=0).fit(X_train,Y_train)
dtscore=dtmodel.score(X,Y)
Y_pred = dtmodel.predict(X_test)
dtaccuracy=metrics.accuracy_score(Y_test,Y_pred)