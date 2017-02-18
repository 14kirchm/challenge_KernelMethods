import numpy as np
from numpy import linalg
import cvxopt #https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
import pylab as pl
import matplotlib.pyplot as plt
import os.path
from numpy import save, load

path_to_data = '../data/'
path_to_fig = '../figures/'

def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM():
    def __init__(self, kernel, C):
        self.kernel = kernel
        self.C = C

    def fit(self, X, Y):
        number_samples, number_features = X.shape

        # Gram matrix
        K = np.zeros((number_samples, number_samples))
        for i in range(number_samples):
            for j in range(number_samples):
                K[i,j] = self.kernel(X[i], X[j])

        # solve QP
        P = cvxopt.matrix(np.outer(Y,Y) * K,tc='d')
        q = cvxopt.matrix(-1*np.ones(number_samples),tc='d')
        A = cvxopt.matrix(np.resize(Y, (1,number_samples)),tc='d')
        b = cvxopt.matrix(0.0,tc='d')
        G = cvxopt.matrix(np.vstack((np.eye(number_samples), -np.eye(number_samples))),tc='d')
        h = cvxopt.matrix(np.hstack((np.ones(number_samples) * self.C, np.zeros(number_samples))),tc='d')

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha = np.ravel(solution['x'])

        # Support vectors correspond to non-zero lagrange multipliers
        sv = (alpha > 1e-5)
        self.alpha = alpha[sv]
        self.support_vectors_x = X[sv]
        self.support_vectors_y = Y[sv]
        print("%d support vectors for %d training points" % (len(self.alpha), number_samples))

        # Intercept
        self.b = 0
        indices = np.arange(len(alpha))[sv] # non-zero lagrange multipliers indices
        for i in range(len(self.alpha)):
            self.b += self.support_vectors_y[i]
            self.b -= np.sum(self.alpha * self.support_vectors_y * K[indices[i],sv])
        self.b /= len(self.alpha)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(number_features)
            for i in range(len(self.alpha)):
                self.w += self.alpha[i] * self.support_vectors_y[i] * self.support_vectors_x[i] #sum_i alpha_i y_i x_i
        else:
            self.w = None

    def score(self, X):
        if self.kernel == linear_kernel:
            return np.dot(X, self.w) + self.b
        else:
            Y_predict = np.zeros(len(X))
            for i in range(len(X)):
                Y_predict[i] = 0
                for alpha, support_vectors_x, support_vectors_y in zip(self.alpha, self.support_vectors_x, self.support_vectors_y):
                    Y_predict[i] += alpha * support_vectors_y * self.kernel(X[i], support_vectors_x)
            return Y_predict + self.b

    def predict(self, X):
        return np.sign(self.score(X))

    #en dimension 2 uniquement
    def save_plot(self, X_train, Y_train, i):
        X1_train = X_train[Y_train == 1]
        X2_train = X_train[Y_train == -1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        legend = []
        ax.plot(X1_train[:,0], X1_train[:,1], "ro")
        ax.plot(X2_train[:,0], X2_train[:,1], "bo")
        ax.scatter(self.support_vectors_x[:,0], self.support_vectors_x[:,1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = self.predict(X).reshape(X1.shape)
        ax.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        ax.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        ax.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        legend.append('Classifier %d' %i)
        plt.savefig(path_to_fig + str(i) + '.eps', format='eps', dpi=1000)
        plt.close()

def one_vs_all(X,y,erase=True):

    if os.path.isfile(path_to_data + 'parameters.npy') and not erase:
        parameters = load(path_to_data + 'parameters.npy').item()
        return parameters

    number_of_classes=len(np.unique(y))
    parameters={}

    for i in range(number_of_classes):
        print ("Class %d/%d" %(i+1,number_of_classes))
        y_binary=np.array([1 if label == i else -1 for label in y])
        svm = SVM(gaussian_kernel,C=4)
        svm.fit(X,y_binary)
        parameters[i]=svm
        #parameters[i].save_plot(X,y,i)

    save(path_to_data + 'parameters.npy', parameters)

    return parameters

def predict_multiclass(X,number_of_classes,parameters):
    decision_function=np.zeros((X.shape[0],number_of_classes))

    for i in range(number_of_classes):
        decision_function[:,i]=np.absolute(parameters[i].score(X))

    y_hat = np.argmax(decision_function,axis=1)

    return y_hat