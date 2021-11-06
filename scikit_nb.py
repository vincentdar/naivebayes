# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-11-06 11:57:30
# @Last Modified by:   Your name
# @Last Modified time: 2021-11-06 12:05:08

#!/usr/bin/env python
# coding: utf-8

# # Scikit-learn Gaussian Naive-Bayes HDI implementation

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

raw_data = pd.read_csv('Development Index.csv')
raw_data.describe()

df = raw_data.drop(columns=['Pop. Density '])
X, y = df.iloc[:, 0:-1], df.iloc[:, -1:]

X = X.to_numpy()
y = y.to_numpy().squeeze()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)


y_true = y_test

target_names = ['1', '2', '3', '4']
print(classification_report(y_true, y_pred, target_names=target_names))







