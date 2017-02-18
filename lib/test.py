import numpy as np
from svm import *

# Saves the figure in /figures

def gen_lin_separable_overlap_data():
    # generate training data in the 2-d case
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def split_train(X1, y1, X2, y2):
    X1_train = X1[:90]
    y1_train = y1[:90]
    X2_train = X2[:90]
    y2_train = y2[:90]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train

def split_test(X1, y1, X2, y2):
    X1_test = X1[90:]
    y1_test = y1[90:]
    X2_test = X2[90:]
    y2_test = y2[90:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test

def test_soft():
    X1, Y1, X2, Y2 = gen_lin_separable_overlap_data()
    X_train, Y_train = split_train(X1, Y1, X2, Y2)
    X_test, Y_test = split_test(X1, Y1, X2, Y2)

    print X_train

    clf = SVM(gaussian_kernel,C=1000.1)
    clf.fit(X_train, Y_train)

    Y_predict = clf.predict(X_test)
    correct = np.sum(Y_predict == Y_test)
    print("%d out of %d predictions correct" % (correct, len(Y_predict)))

    clf.save_plot(X_train, Y_train,1)

test_soft()
