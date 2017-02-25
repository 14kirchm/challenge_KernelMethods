from __future__ import division
import pandas as pd
import numpy as np
import math
import time

from svm import *

path_to_results = '../results/'
path_to_data = '../data/'

X_test = pd.read_csv(path_to_data + 'Xte.csv', header=None, usecols=range(3072))
Y_train = pd.read_csv(path_to_data + 'Ytr.csv')
X_train = pd.read_csv(path_to_data + 'Xtr.csv', header=None, usecols=range(3072))

#size_training = 1000
#size_test = 100

######## Parameters #########
C = 10
kernel = polynomial_kernel
#############################

"""
# delete last column (NaN)
X_tot_tmp = X_train.values#[:size_training]
X_tot = np.zeros((X_tot_tmp.shape[0], X_tot_tmp.shape[1]-1))
for i in range(len(X_tot)):
    X_tot[i] = X_tot_tmp[i][:-1]
"""

"""
# X_tot_test = X_test.values
X_tot_test_tmp = X_train.values[-size_test:]
# delete last column (NaN)
X_tot_test = np.zeros((X_tot_test_tmp.shape[0], X_tot_test_tmp.shape[1]-1))
for i in range(len(X_tot_test)):
    X_tot_test[i] = X_tot_test_tmp[i][:-1]
Y_tot_test = Y_train["Prediction"].values[-size_test:]
"""

X_tot, Y_tot, X_tot_test, Y_tot_test = split(X_train, Y_train)

Xtrain = X_tot.values
Ytrain = Y_tot["Prediction"].values
Xval = X_tot_test.values
Yval = Y_tot_test["Prediction"].values

print("Fitting")
parameters = one_vs_all(Xtrain,Ytrain, C, )

print('____________________')

print("Predicting")
temps = time.time()

number_of_classes = len(np.unique(Ytrain))
result = predict_multiclass(Xval, number_of_classes, parameters)
print("Results: ")
print(result)
print("Groundtruth: ")
print(Yval)
precision = np.sum(Yval == result)/len(result)
print("Precision: %f" % precision)

print("Time: ", time.time() - temps)
print("Number of observations: ", X_test.values.shape)  # [0])

print('____________________')

print("Saving results")

f = open(path_to_results+"result.txt", "w")

f.write("Id,Prediction\n")
for i, a in enumerate(result):
    f.write(str(i+1))
    f.write(",")
    f.write(str(a))
    f.write("\n")

f.close()
