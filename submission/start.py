from __future__ import division
import pandas as pd
import numpy as np
import math
import time

from features import hog
from svm import *
from rotations import rotate_bdd
from rotations import flip_bdd

path_to_results = '../results/'
path_to_data = '../data/'

######## Parameters #########
C = .1
kernel = gaussian_kernel
size_training = 5000  # Max 5000
valid_ratio = -0.2
number_folds = 1
#############################

X_test = pd.read_csv(path_to_data + 'Xte.csv', header=None, usecols=range(3072))
Y_train_0 = pd.read_csv(path_to_data + 'Ytr.csv', nrows=size_training)
X_train = pd.read_csv(path_to_data + 'Xtr.csv', header=None, usecols=range(3072), nrows=size_training)

HOG_train = np.vstack([rotate_bdd(X_train, 0), rotate_bdd(X_train[:500], 10), rotate_bdd(X_train[:500], -10), flip_bdd(X_train[:500])])
Y_train_1 = pd.read_csv(path_to_data + 'Ytr.csv', nrows=size_training)
Y_train_1.index += size_training
Y_train_2 = pd.read_csv(path_to_data + 'Ytr.csv', nrows=size_training)
Y_train_2.index += size_training + 500
Y_train_3 = pd.read_csv(path_to_data + 'Ytr.csv', nrows=size_training)
Y_train_3.index += size_training + 2*500
Y_train = pd.concat([Y_train_0, Y_train_1[:500], Y_train_2[:500], Y_train_3[:500]])

#Cross Validation if valid_ratio > 0
if (valid_ratio > 0):
    for n in range(number_folds):
        # Split data
        n = len(Y_train)
        end = int(valid_ratio * n)
        perm = np.random.permutation(Y_train.index)
        Xval = HOG_train[perm[0:end], :]
        Yval = Y_train['Prediction'][perm[0:end]].as_matrix()
        Xtrain = HOG_train[perm[end:], :]
        Ytrain = Y_train['Prediction'][perm[end:]].as_matrix()

        print("Fitting")
        parameters = one_vs_all(Xtrain, Ytrain, C, kernel)
        print('____________________')

        print("Predicting")
        temps = time.time()

        number_of_classes = len(np.unique(Ytrain))
        result_CV = predict_multiclass(Xval, number_of_classes, parameters)

        print("Results: ")
        print(result_CV)
        print("Groundtruth: ")
        print(Yval)
        precision = np.sum(Yval == result_CV)/len(result_CV)
        print("Precision: %f" % precision)

        print("Time: ", time.time() - temps)
        print('____________________')

# Compute HOG vector on testing set
HOG_test = np.zeros((len(X_test), 324))
for i in range(len(X_test)):
    im = X_test.ix[i].values.reshape(3, 32, 32).transpose(1, 2, 0)
    HOG_test[i, :] = hog(im)

Xtrain = HOG_train
Ytrain = Y_train['Prediction'].as_matrix()
Xtest = HOG_test

print("Fitting")
parameters = one_vs_all(Xtrain, Ytrain, C, kernel)
print('____________________')

print("Predicting")
temps = time.time()

number_of_classes = len(np.unique(Ytrain))
result = predict_multiclass(Xtest, number_of_classes, parameters)

print("Time: ", time.time() - temps)
print('____________________')

print("Saving results")

f = open(path_to_results+"Yte.csv", "w")

f.write("Id,Prediction\n")
for i, a in enumerate(result):
    f.write(str(i+1))
    f.write(",")
    f.write(str(a))
    f.write("\n")

f.close()
