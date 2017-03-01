# rotate training images
from skimage.feature import hog
import skimage.io
from skimage.transform import rotate
import numpy as np

from features import hog

def rotate_bdd(X_train,deg):
    HOG_train = np.zeros((len(X_train), 324))
    for i in range(len(X_train)):
        image = X_train.ix[i].values.reshape(3, 32, 32).transpose(1, 2, 0)
        f0 = rotate(image[:,:,0], deg, resize=True)
        f1 = rotate(image[:,:,1], deg, resize=True)
        f2 = rotate(image[:,:,2], deg, resize=True)
        im = np.rollaxis(np.dstack((f0, f1, f2)), 2, 0).transpose(1, 2, 0)
        HOG_train[i, :] = hog(im)
    return HOG_train
