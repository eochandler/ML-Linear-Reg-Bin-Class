#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:07:27 2020

@author: elijahchandler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model
import sklearn.neighbors

df2 = pd.read_csv('/Users/elijahchandler/Desktop/ieee-fraud-detection/train_transaction.csv')

Y_train = df2['isFraud']

def prepareFraudData(df):
    cols = ['TransactionAmt', 'card4', 'card6']
    X = df[cols]
    
    oneHot = pd.get_dummies(X['card4'])
    X = X.drop(['card4'], axis = 1)
    X = X.join(oneHot)
    
    oneHot2 = pd.get_dummies(X['card6'])
    X = X.drop(['card6'], axis = 1)
    X = X.join(oneHot2)
    
    X['TransactionAmt'] = X['TransactionAmt'].fillna(X['TransactionAmt'].mean())
    
    X = (X-X.mean())/X.std()
    return X

X_train = prepareFraudData(df2)

df_test = pd.read_csv('/Users/elijahchandler/Desktop/ieee-fraud-detection/test_transaction.csv')
X_test = prepareFraudData(df_test)
X_test['debit or credit'] = 0

#need to implement my down logistic regression 

df2_fraud = df2[df2['isFraud'] ==1]
Y_train_fraud = Y_train[df2['isFraud'] == 1 ]
df2_notFraud = df2[df2['isFraud'] == 0 ]

df2_notFraud = df2_notFraud[0:20000]
Y_train_notFraud = Y_train[df2['isFraud'] == 0 ]
Y_train_notFraud = Y_train_notFraud[0:20000]

df2_reduced = pd.concat([df2_fraud, df2_notFraud], ignore_index = True)
Y_train_reduced = pd.concat([Y_train_fraud, Y_train_notFraud], ignore_index = True)

X_train_reduced= prepareFraudData(df2_reduced)
X_train_reduced['charge card'] = 0

###########################
#Log reg function from scratch
def logReg(X, Y):
    N, d = X.shape
    allOnes = np.ones((N, 1))
    X = np.hstack((allOnes, X))
    
    alpha = 0.0001
    beta = np.random.randn(d+1)
    listNormGrad = []
    listLBetas = []
    
    #Change to 300 or 500 to check if it has effect on solution
    maxiterations = 30
    plt.figure()
    for idx in range(maxiterations):
        gradient = np.zeros(d+1)
        
        for i in range(N):
            Xi = X[i, :]
            Yi = Y[i]
            qi = sigmoid(np.vdot(Xi, beta))
            
            gradient += (qi - Yi) * Xi
            
            
        norm_gradient = np.linalg.norm(gradient)
        listNormGrad.append(norm_gradient)
        beta = beta - alpha * gradient  
        LBeta = L(beta, X, Y)
        listLBetas.append(LBeta)
        print(idx, LBeta)
    return beta, listNormGrad, listLBetas

def sigmoid(u):
    return np.exp(u)/(1+ np.exp(u))


def L(beta, X, Y):
    N = X.shape[0]
    mySumHi = 0
    for i in range(N):
        xihat = X[i]
        yi = Y[i]
        dotProduct = np.vdot(xihat, beta)
        mySumHi += h(dotProduct, yi)
    return mySumHi


def h(u, yi):
    exp = np.exp(u)
    return -yi * u + np.log(1 + exp)

beta,listNormGrad,listLBetas = logReg(X_train_reduced.values[0:5000], Y_train_reduced.values[0:5000])

#%%
N, d = X_test.shape
allOnes = np.ones((N, 1))
X_test = np.hstack((allOnes, X_test))

#X_test.values is a matrix with feaure vectors 'dot product' vector beta~500,000 times
dotproducts = X_test @beta
predictions = sigmoid(dotproducts)

#parameter vector beta contains all predictors

kaggleSubmission = df_test[['TransactionID']]
kaggleSubmission['isFraud'] = predictions
kaggleSubmission.to_csv('/Users/elijahchandler/Desktop/mySubmission.csv',index = False)