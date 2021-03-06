import numpy as np
import cv2
from scipy import ndimage

def get_neighbors(x, y):
    # neighbor window = 41*41, sampled with spacing = 5
    neighbor = np.indices((9, 9)) - 4
    neighbor = neighbor*5
    neighbor[0] += x
    neighbor[1] += y
    return np.reshape(neighbor, (2, 81, 1))

    #return np.array([(x-1, y-1), (x, y-1), (x+1, y-1), (x-1, y), (x, y), (x+1, y), (x-1, y+1), (x, y+1), (x+1, y+1)])

def descriptor(img, xy):
    features = np.zeros((len(xy), 81))
    for i in range(len(xy)):
        neighbors = get_neighbors(xy[i, 0], xy[i, 1])
        for j in range(81):
            features[i, j] = img[neighbors[1, j], neighbors[0, j]]
    return features

def findMinArg(xy_1, features_1, xy_2, features_2):
    arg = {} 
    for f in range(len(features_2)):
        diff = np.tile(features_2[f], (len(features_1),1)) - features_1
        diff = diff**2
        diff = np.sum(diff, 1)
        min_arg = np.argmin(diff)
        arg[f] = min_arg
    return arg
     
def find_pair_2(xy_1, features_1, xy_2, features_2):
        arg_2 = findMinArg(xy_1, features_1, xy_2, features_2)
        arg_1 = findMinArg(xy_2, features_2, xy_1, features_1)
        pair_1 = []
        pair_2 = []

        for i in arg_1:
            if arg_1[i] not in arg_2:
                continue;
            if arg_2[arg_1[i]] == i:
                pair_1.append(xy_1[i])
                pair_2.append(xy_2[arg_1[i]])
        
        return np.array([pair_1, pair_2], dtype='uint32')

def find_pair(xy_1, features_1, xy_2, features_2):
    value = {}
    arg = {}
    pair = []
    for f in range(len(features_2)):
        diff = np.tile(features_2[f], (len(features_1),1)) - features_1
        diff = diff**2
        diff = np.sum(diff, 1)
        min_arg = np.argmin(diff)
        min_value = np.min(diff)
        pair.append(min_arg)
        if min_arg in value:
            if not (min_value < value[min_arg]):
                continue
        arg[min_arg] = f
        value[min_arg] = min_value
    pair_1 = np.zeros((len(arg),2), dtype = 'uint32')
    pair_2 = np.zeros((len(arg),2), dtype = 'uint32')
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
    A = np.zeros((len(pair_2)*2, 3))
    B = np.zeros((len(pair_1)*2, 2))
    for i in range(len(pair_1)):
        B[2*i, 0] = pair_1[i, 0]
        B[2*i, 1] = pair_1[i, 1]
        A[2*i, 0] = pair_2[i, 0]
        A[2*i, 1] = pair_2[i, 1]
        A[2*i, 2] = 1.0

    x = np.zeros((3,2))
    err, x = cv2.solve(A, B, x, cv2.DECOMP_SVD)
    return np.transpose(x)

def drawMatchLine(img_1, img_2, pair_1, pair_2):
    rows, cols, ch = img_1.shape
    img = np.concatenate((img_1, img_2), axis=1)
    img = np.array(img)
    for i in range(len(pair_1)):
        pt1 = (pair_1[i][0], pair_1[i][1])
        pt2 = (pair_2[i][0]+cols, pair_2[i][1])
        cv2.line(img, pt1, pt2, (0,0,255))
    return img

def main():
    xy_1 = np.random.randint(5, size = (10,2))+2
    xy_2 = np.random.randint(5, size = (10,2))+2
    print xy_1
    print xy_2
    img = np.random.random((15,15))
    features_1 = descriptor(img, xy_1)
    features_2 = descriptor(img, xy_2)
    pair_1, pair_2 = find_pair(xy_1, features_1, xy_2, features_2)
    print pair_1
    print pair_2
    x = solve_M(pair_1, pair_2)
    

if __name__ == '__main__':
    main() 
