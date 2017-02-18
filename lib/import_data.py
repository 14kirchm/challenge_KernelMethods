import csv
import numpy as np
import matplotlib.pyplot as plt

Xte = np.zeros((2000, 32, 32, 3))

with open('data/Xte.csv') as fid:
    f = csv.reader(fid)
    i = 0
    for row in f:
        row_float = [float(k) for k in row[:-1]]
        Xte[i, :, :, :] = np.reshape(row_float, (32, 32, 3), order='F')
        plt.imshow(Xte[i, :, :, :])
        plt.show()
        i += 1
        import pdb; pdb.set_trace()
