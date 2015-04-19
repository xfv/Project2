import numpy as np
import math
import cv2

def assemble(img_bgr, M):
    corners = []
    descriptors = []
    pairs = []
    M = []
    for i in range(len(img_bgr)):
        img_y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        corners.append(harris(img_y))
        descriptors.append(descriptor(img_y, corners[i]))
    for i in range(len(img_bgr)-1):
        pairs.append(find_pair(corners[i], descriptors[i], corners[i+1], descriptors[i+1]))
        M.append(solve_M(pairs[i][0], pairs[i][1]))
    
    rows, cols = img_bgr[0]
    # Handling panorama boudaries
    x_max = cols
    y_min = 0
    y_max = rows
    
    position = []
    for i in range(len(img_bgr)):
        x = np.array(range(cols))
        y = np.array(range(rows))
        x = np.reshape(x, (rows, cols))
        y = np.reshape(y, (rows, cols))
        if i == 0:
            position.append([x, y])
            continue
        new_x = x*M[i-1][0,0] + y*M[i-1][0,1] # x*m11 + y*m12 = x'
        new_y = x*M[i-1][1,0] + y*M[i-1][1,1] # x*m21 + y*m22 = y'
        if new_x[0, -1] > x_max:
            x_max = new_x[0, -1]
        if new_y[0, 0] < y_min:
            y_min = new_y[0, 0]
        if new_y[-1, -1] > y_max:
            y_max = new_y[-1, -1]
        position.append([new_y, new_x])

    panorama = np.zeros((rows, cols, 3))
    panorama[:, (-1)*y_min:(-1)*y_min+rows ,:] = img_bgr[0]
    for i in range(1, len(img_bgr)):
        y_begin = position[i][0][0,0] 
        y_end = y_begin + rows
        x_begin = position[i][1][0,0]
        x_end = x_begin + cols
        panorama[y_begin:y_end, x_begin:x_end, :] += img_bgr[i]
        if position[i][0][0,0] > position[i-1][0][0,0]:
            weight_y_start = position[i-1][0][0,0]
            weight_y_end = position[i][0][-1,0] + 1
        else if position[i][0][0,0] < position[i-1][0][0,0]:
            weight_y_start = position[i][0][0,0]
            weight_y_end = position[i-1][0][-1,-1] + 1
        else:
            weight_y_start = position[i][0][0,0]
            weight_y_end = position[i][-1,0] + 1
        weight_x_start = position[i][1][0,0]
        weight_x_end = position[i-1][1][-1,-1] + 1
        if weight_y_start == weight_y_end or weight_x_start == weight_x_end:
            print 'WARNING!!! NO OVERLAP!!!'
        panorama[weight_y_start:weight_y_end, weight_x_start:weight_x_end, :] /= 2.0   
    cv2.imwrite('panorama.jpg', panorama) 


