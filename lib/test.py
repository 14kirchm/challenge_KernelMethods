import numpy as np
from svm import *

# Saves the figure in /figures
path_to_fig = '../figures/'

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

    clf = SVM(gaussian_kernel, C=1)
    clf.fit(X_train, Y_train)

    Y_predict = clf.predict(X_test)
    correct = np.sum(Y_predict == Y_test)
    print("%d out of %d predictions correct" % (correct, len(Y_predict)))

    clf.save_plot(X_train, Y_train,3)

def test_soft_2():
    n_samples = 30
    mean_1 = [0, -3]
    cov = [[3, 0], [0, 3]]
    X_1 = np.random.multivariate_normal(mean_1, cov, n_samples).T
    mean_2 = [3, 3]
    X_2 = np.random.multivariate_normal(mean_2, cov, n_samples).T
    mean_3 = [-3, 3]
    X_3 = np.random.multivariate_normal(mean_3, cov, n_samples).T
    X = np.concatenate((X_1, X_2, X_3), axis = 1)
    X = np.concatenate((np.ones((1, 3*n_samples)), X), axis=0)
    X = X.T

    y = np.concatenate((np.zeros((1,n_samples)), np.ones((1, n_samples)), 2*np.ones((1,n_samples))), axis=1)
    y = y[0,:]

    parameters = one_vs_all(X, y, 1, gaussian_kernel, True, False)

    y_pred = predict_multiclass(X, 3, parameters)

    plt.figure()
    plt.plot(X[y==0,1:][:,0], X[y==0,1:][:,1], '+')
    plt.plot(X[y==1,1:][:,0], X[y==1,1:][:,1], 'o')
    plt.plot(X[y==2,1:][:,0], X[y==2,1:][:,1], '.')
    #plt.plot(X[:,0], X[:,1], 'o')
    plt.savefig(path_to_fig + str(1) + '.eps', format='eps', dpi=1000)
    plt.close()

    plt.figure()
    plt.plot(X[y_pred==0,1:][:,0], X[y_pred==0,1:][:,1], '+')
    plt.plot(X[y_pred==1,1:][:,0], X[y_pred==1,1:][:,1], 'o')
    plt.plot(X[y_pred==2,1:][:,0], X[y_pred==2,1:][:,1], '.')
    plt.savefig(path_to_fig + str(2) + '.eps', format='eps', dpi=1000)
    plt.close()

test_soft_2()

test_soft()
