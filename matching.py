import numpy as np
import cv2
from scipy import ndimage

def get_neighbors(x, y):
    return np.array([(x-1, y-1), (x, y-1), (x+1, y-1), (x-1, y), (x, y), (x+1, y), (x-1, y+1), (x, y+1), (x+1, y+1)])

def descriptor(img, xy):
    features = np.zeros((len(xy), 9))
    for i in range(len(xy)):
        neighbors = get_neighbors(xy[i, 0], xy[i, 1])
        for j in range(9):
            features[i, j] = img[neighbors[j, 0], neighbors[j, 1]]
    return features


def solve_M(xy_1, features_1, xy_2, features_2):
    value = {}
    pair = []
    for f in range(len(features_2)):
        diff = np.tile(features_2[f], (len(features_1),1)) - features_1
        diff = diff**2
        diff = sum(diff, 1)
        min_arg = np.argmax(diff)
        min_value = np.min(diff)
        if min_arg in value:
            if min_value > value[min_arg]:
                continue
        value[min_arg] = min_value
        pair.append(min_arg)

    # solving Ax = B
    A = np.zeros((len(pair)*2, 4))
    B = np.zeros((len(pair)*2, 1))
    for i in range(len(pair)):
        B[2*i] = xy_1[pair[i], 0]
        B[2*i+1] = xy_1[pair[i], 1]
        A[2*i, 0] = xy_2[i, 0]
        A[2*i, 1] = xy_2[i, 1]
        A[2*i+1, 2] = xy_2[i, 0]
        A[2*i+1, 3] = xy_2[i, 1]

    x = np.zeros((4,1))
    err, x = cv2.solve(A, B, x, cv2.DECOMP_SVD)
    return x

xy_1 = np.random.randint(10, size = (10,2))
xy_2 = np.random.randint(10, size = (10,2))
img = np.random.random((15,15))
features_1 = descriptor(img, xy_1)
features_2 = descriptor(img, xy_2)
x = solve_M(xy_1, features_1, xy_2, features_2)
