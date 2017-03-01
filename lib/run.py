from __future__ import division
import pandas as pd
import numpy as np
import math
import time

# from skimage.feature import hog
from features import hog
from svm import *
from rotations import rotate_bdd

path_to_results = '../results/'
path_to_data = '../data/'

######## Parameters #########
C = .1
kernel = gaussian_kernel
size_training = 1000  # Max 5000
valid_ratio = 0.3
number_folds = 1
#############################

X_test = pd.read_csv(path_to_data + 'Xte.csv', header=None, usecols=range(3072))
Y_train_0 = pd.read_csv(path_to_data + 'Ytr.csv', nrows=size_training)
X_train = pd.read_csv(path_to_data + 'Xtr.csv', header=None, usecols=range(3072), nrows=size_training)

"""
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
"""

HOG_train = np.vstack([rotate_bdd(X_train,0),rotate_bdd(X_train,10),rotate_bdd(X_train,-10)])
Y_train_1 = pd.read_csv(path_to_data + 'Ytr.csv', nrows=size_training)
Y_train_2 = pd.read_csv(path_to_data + 'Ytr.csv', nrows=size_training)
Y_train_1.index += size_training
Y_train_2.index += 2*size_training
Y_train = pd.concat([Y_train_0,Y_train_1,Y_train_2])

"""
# Compute HOG vector on training set
HOG_train = np.zeros((len(X_train), 324))
for i in range(len(X_train)):
    im = X_train.ix[i].values.reshape(3, 32, 32).transpose(1, 2, 0)

    # imGray = rgb2gray(im)
    # HOG_train[i, :] = hog(imGray, cells_per_block=(2,2))
    HOG_train[i, :] = hog(im)
"""

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
    # imGray = rgb2gray(im)
    # HOG_test[i, :] = hog(imGray, cells_per_block=(2,2))
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

f = open(path_to_results+"result.txt", "w")

f.write("Id,Prediction\n")
for i, a in enumerate(result):
    f.write(str(i+1))
    f.write(",")
    f.write(str(a))
    f.write("\n")

f.close()










"""
FIXED CV

size_training = 1000
size_test = 100

X_test = pd.read_csv(path_to_data + 'Xte.csv', header=None)
Y_train = pd.read_csv(path_to_data + 'Ytr.csv')
X_train = pd.read_csv(path_to_data + 'Xtr.csv', header=None)
# delete last column (NaN)
X_tot_tmp = X_train.values[:size_training]
X_tot = np.zeros((X_tot_tmp.shape[0], X_tot_tmp.shape[1]-1))
for i in range(len(X_tot)):
    X_tot[i] = X_tot_tmp[i][:-1]
Y_tot = Y_train["Prediction"].values[:size_training]

X_tot_test_tmp = X_train.values[-size_test:]
# delete last column (NaN)
X_tot_test = np.zeros((X_tot_test_tmp.shape[0], X_tot_test_tmp.shape[1]-1))
for i in range(len(X_tot_test)):
    X_tot_test[i] = X_tot_test_tmp[i][:-1]
Y_tot_test = Y_train["Prediction"].values[-size_test:]

"""

"""
VARIABLE CV NON HOG

X_tot, Y_tot, X_tot_test, Y_tot_test = split(X_train, Y_train)

Xtrain = X_tot.values
Ytrain = Y_tot["Prediction"].values
Xval = X_tot_test.values
Yval = Y_tot_test["Prediction"].values
"""
