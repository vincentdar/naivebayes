# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-11-07 13:06:52
# @Last Modified by:   Your name
# @Last Modified time: 2021-11-07 13:14:50
#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Classifier from scratch 


from math import exp, pi, sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import norm
from math import log
import numpy as np
import pandas as pd


def probability(X, prior, dists):
    prob = prior
    idx = 0
    for dist in dists:
        tmp = dist.pdf(X[idx])
        if tmp <= 0:
            tmp = 0.1
        res = log(tmp)
        
        prob = prob + res
        idx = idx + 1
    
    return prob

def fit_distribution(data):
    mean = data.mean()
    sigma = data.std()
    # print("Mean:", mean, "Sigma:", sigma)
    dist = norm(mean, sigma)
    return dist

def predict(data):
    category_ls = []
    category_ls.append(probability(data, priory1, [X1y1, X2y1, X3y1, X4y1 ,X5y1]))
    category_ls.append(probability(data, priory2, [X1y2, X2y2, X3y2, X4y2 ,X5y2]))
    category_ls.append(probability(data, priory3, [X1y3, X2y3, X3y3, X4y3 ,X5y3]))
    category_ls.append(probability(data, priory4, [X1y4, X2y4, X3y4, X4y4 ,X5y4]))
    
    maximum = np.argmax(category_ls)
    return maximum + 1

if __name__ == '__main__':
    raw_data = pd.read_csv('Development Index.csv')

    # ## Drop unnecessary column
    df = raw_data.drop(columns=['Pop. Density '])
    df = df.astype('float32')

    # ## Sort the data into classes
    y_true = df.iloc[:, -1:]
    # ## Split the training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:], df.iloc[:, -1], test_size=0.2, random_state=0)

    Xy1 = X_train.loc[(X_train["Development Index"] == 1)]
    Xy2 = X_train.loc[(X_train["Development Index"] == 2)]
    Xy3 = X_train.loc[(X_train["Development Index"] == 3)]
    Xy4 = X_train.loc[(X_train["Development Index"] == 4)]
    Xy1.shape, Xy2.shape, Xy3.shape, Xy4.shape


    # ## Calculate Priors
    priory1 = len(Xy1) / len(X_train)
    priory2 = len(Xy2) / len(X_train)
    priory3 = len(Xy3) / len(X_train)
    priory4 = len(Xy4) / len(X_train)
    priory1, priory2, priory3, priory4


    # Create PDF HDI == 1
    X1y1 = fit_distribution(Xy1.iloc[:, 0])
    X2y1 = fit_distribution(Xy1.iloc[:, 1])
    X3y1 = fit_distribution(Xy1.iloc[:, 2])
    X4y1 = fit_distribution(Xy1.iloc[:, 3])
    X5y1 = fit_distribution(Xy1.iloc[:, 4])


    # Create PDF HDI == 2
    X1y2 = fit_distribution(Xy2.iloc[:, 0])
    X2y2 = fit_distribution(Xy2.iloc[:, 1])
    X3y2 = fit_distribution(Xy2.iloc[:, 2])
    X4y2 = fit_distribution(Xy2.iloc[:, 3])
    X5y2 = fit_distribution(Xy2.iloc[:, 4])


    # Create PDF HDI == 3
    X1y3 = fit_distribution(Xy3.iloc[:, 0])
    X2y3 = fit_distribution(Xy3.iloc[:, 1])
    X3y3 = fit_distribution(Xy3.iloc[:, 2])
    X4y3 = fit_distribution(Xy3.iloc[:, 3])
    X5y3 = fit_distribution(Xy3.iloc[:, 4])


    # Create PDF HDI == 4
    X1y4 = fit_distribution(Xy4.iloc[:, 0])
    X2y4 = fit_distribution(Xy4.iloc[:, 1])
    X3y4 = fit_distribution(Xy4.iloc[:, 2])
    X4y4 = fit_distribution(Xy4.iloc[:, 3])
    X5y4 = fit_distribution(Xy4.iloc[:, 4])

    # ## Prepare to test and evaluate
    y_true = X_test.iloc[:, -1]
    X_test, y_true



    res = [predict(x) for x in X_test.iloc[:, :-1].to_numpy()]
    res_np = np.array(res)
    res_np, len(res_np)


    y_true_np = y_true.to_numpy()
    y_true_np = y_true_np.astype('int64')
    y_true_np, len(y_true)
    target_names = ['1', '2', '3', '4']

    print(classification_report(y_true_np, res_np, target_names=target_names))


    # correct = 0
    # incorrect = 0
    # idx = 0
    # for item in y_true_np:
    #     if item == res_np[idx]:
    #         correct += 1
    #     else:
    #         incorrect += 1
    #     idx += 1

    # print("Correct:", correct, "Incorrect:", incorrect, "Total:", correct + incorrect)

