# rotate training images
from features import hog
import numpy as np

def flip(image, dim=0):
    image_flipped = np.zeros(image.shape)
    if(dim == 0):
        image_flipped = image[::-1, :, :]
    elif(dim == 1):
        image_flipped = image[:, ::-1, :]
    else:
        image_flipped = image
    return image_flipped

def rotate(im, deg):
    U, V, nc = im.shape
    u_center, v_center = (U-1)/2, (V-1)/2
    cos = np.cos(np.deg2rad(deg))
    sin = np.sin(np.deg2rad(deg))
    indices_u = cos * (np.indices((U, V))[0] - u_center) + sin * (np.indices((U, V))[1] - v_center)
    indices_v = -sin * (np.indices((U, V))[0] - u_center) + cos * (np.indices((U, V))[1] - v_center)
    im_rotated = np.zeros(im.shape)
    for u in range(U):
        for v in range(V):
            d = (u-u_center - indices_u) ** 2 + (v-v_center - indices_v) ** 2
            if (np.amin(d) < 1):
                u0, v0 = np.unravel_index(np.argmin(d), (U, V))
                im_rotated[u, v, :] = im[u0, v0, :]
    return im_rotated

def rotate_bdd(X_train, deg):
    HOG_train = np.zeros((len(X_train), 324))
    for i in range(len(X_train)):
        image = X_train.ix[i].values.reshape(3, 32, 32).transpose(1, 2, 0)
        im = rotate(image, deg)
        HOG_train[i, :] = hog(im)
    return HOG_train

def flip_bdd(X_train):
    HOG_train = np.zeros((len(X_train), 324))
    for i in range(len(X_train)):
        image = X_train.ix[i].values.reshape(3, 32, 32).transpose(1, 2, 0)
        im = flip(image, 1)
        HOG_train[i, :] = hog(im)
    return HOG_train
