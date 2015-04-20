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
            features[i, j] = img[neighbors[j, 1], neighbors[j, 0]]
    return features

def find_pair(xy_1, features_1, xy_2, features_2):
    value = {}
    arg = {}
    pair = []
    for f in range(len(features_2)):
        diff = np.tile(features_2[f], (len(features_1),1)) - features_1
        diff = diff**2
        diff = sum(diff, 1)
        print 'f', f
        print 'diff', diff
        min_arg = np.argmin(diff)
        min_value = np.min(diff)
        pair.append(min_arg)
        if min_arg in value:
            if not (min_value < value[min_arg]):
                continue
        arg[min_arg] = f
        value[min_arg] = min_value
    pair_1 = np.zeros((len(arg),2), dtype='uint8')
    pair_2 = np.zeros((len(arg),2), dtype='uint8')
    pair_idx = 0
    for p in range(len(pair)):
        if not arg[pair[p]] == p:
            continue
        pair_1[pair_idx] = xy_1[pair[p]]
        pair_2[pair_idx] = xy_2[p]
        pair_idx += 1
    return pair_1, pair_2

def solve_M(pair_1, pair_2):
    # solving Ax = B
    A = np.zeros((len(pair_2)*2, 4))
    B = np.zeros((len(pair_1)*2, 1))
    for i in range(len(pair_1)):
        B[2*i] = pair_1[i, 0]
        B[2*i+1] = pair_1[i, 1]
        A[2*i, 0] = pair_2[i, 0]
        A[2*i, 1] = pair_2[i, 1]
        A[2*i+1, 2] = pair_2[i, 0]
        A[2*i+1, 3] = pair_2[i, 1]

    x = np.zeros((4,1))
    err, x = cv2.solve(A, B, x, cv2.DECOMP_SVD)
    return np.reshape(x, (2,2))

def main():
    xy_1 = np.random.randint(10, size = (10,2))
    xy_2 = np.random.randint(10, size = (10,2))
    print 'xy_1', xy_1
    print 'xy_2', xy_2
    img = np.random.random((15,15))
    features_1 = descriptor(img, xy_1)
    features_2 = descriptor(img, xy_2)
    pair_1, pair_2 = find_pair(xy_1, features_1, xy_2, features_2)
    print 'pair_1', pair_1
    print 'pair_2', pair_2
    x = solve_M(pair_1, pair_2)

if __name__ == '__main__':
    main() 
