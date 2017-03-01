import numpy as np
from numpy import linalg
import cvxopt  # https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
import pylab as pl
import matplotlib.pyplot as plt
import os.path
from numpy import save, load

path_to_data = '../data/'
path_to_fig = '../figures/'


def split(Xtr, Ytr, val_frac=0.3):
    perm = np.random.permutation(Ytr.index)
    print(perm)
    n = len(Ytr)
    end = int(val_frac * n)
    Xval = Xtr.ix[perm[0:end]]
    Yval = Ytr.ix[perm[0:end]]
    Xtrain = Xtr.ix[perm[end:]]
    Ytrain = Ytr.ix[perm[end:]]

    return Xtrain, Ytrain, Xval, Yval


def linear_kernel(x, y):
    return np.dot(x, y)


def polynomial_kernel(x, y, p=4):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=0.8):
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
                K[i, j] = self.kernel(X[i], X[j])

        # solve QP
        P = cvxopt.matrix(K, tc='d')
        q = cvxopt.matrix(-Y, tc='d')
        A = cvxopt.matrix(np.ones((1, number_samples)), tc='d')
        b = cvxopt.matrix(0.0, tc='d')
        G = cvxopt.matrix(np.vstack((np.diag(Y), -np.diag(Y))), tc='d')
        h = cvxopt.matrix(np.hstack((np.ones(number_samples) *
                                    self.C, np.zeros(number_samples))), tc='d')

        # hide outputs
        cvxopt.solvers.options['show_progress'] = False

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha = np.ravel(solution['x'])

        # Support vectors correspond to non-zero lagrange multipliers
        tol = 1e-5
        sv = ((alpha * Y) > tol)
        self.alpha = alpha[sv]
        self.support_vectors_x = X[sv]
        self.support_vectors_y = Y[sv]
        sv_stric = ((alpha * Y) > tol) & ((alpha * Y) < self.C - tol)
        print("%d support vectors for %d training points" % (len(alpha[sv_stric]), number_samples))

        # Intercept
        num_margin_vectors = 0
        self.b = 0
        indices = np.arange(len(alpha))[sv]  # non-zero Lagrange multipliers indices
        for i in range(len(self.alpha)):
            if(self.alpha[i] * self.support_vectors_y[i] < self.C - tol):
                num_margin_vectors += 1
                self.b += self.support_vectors_y[i] - np.dot(self.alpha,K[indices[i], sv])
        self.b /= num_margin_vectors

        # Weight vector (only linear kernel)
        if self.kernel == linear_kernel:
            self.w = np.zeros(number_features)
            for i in range(len(self.alpha)):
                self.w += self.alpha[i] * self.support_vectors_x[i]  #sum_i alpha_i y_i x_i
        else:
            self.w = None

    def score(self, X):
        if self.kernel == linear_kernel:
            return np.dot(X, self.w) + self.b
        else:
            Y_predict = np.zeros(len(X))
            for i in range(len(X)):
                Y_predict[i] = 0
                for alpha, support_vectors_x in zip(self.alpha,self.support_vectors_x):
                    Y_predict[i] += alpha * self.kernel(X[i], support_vectors_x)
            return Y_predict + self.b

    def predict(self, X):
        return np.sign(self.score(X))

    # en dimension 2 uniquement
    def save_plot(self, X_train, Y_train, i):
        X1_train = X_train[Y_train == 1]
        X2_train = X_train[Y_train == -1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        legend = []
        ax.plot(X1_train[:, 0], X1_train[:, 1], "ro")
        ax.plot(X2_train[:, 0], X2_train[:, 1], "bo")
        ax.scatter(self.support_vectors_x[:, 0], self.support_vectors_x[:, 1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = self.predict(X).reshape(X1.shape)
        ax.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        ax.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        ax.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        legend.append('Classifier %d' % i)
        plt.savefig(path_to_fig + str(i) + '.eps', format='eps', dpi=1000)
        plt.close()


def one_vs_all(X, y, C=10, kernel=linear_kernel, erase=True, plot=False):

    if os.path.isfile(path_to_data + 'parameters.npy') and not erase:
        parameters = load(path_to_data + 'parameters.npy').item()
        return parameters

    number_of_classes = len(np.unique(y))
    parameters = {}

    for i in range(number_of_classes):
        print("Class %d/%d" % (i+1, number_of_classes))
        y_class = np.array([1 if label == i else -1 for label in y])
        svm = SVM(kernel, C)
        svm.fit(X, y_class)
        parameters[i] = svm
        if plot:
            parameters[i].save_plot(X,y,i)

    save(path_to_data + 'parameters.npy', parameters)

    return parameters


def predict_multiclass(X, number_of_classes, parameters):
    score = np.zeros((X.shape[0], number_of_classes))

    for i in range(number_of_classes):
        score[:, i] = (parameters[i].score(X))

    y = np.argmax(score, axis=1)

    return y
