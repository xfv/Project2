import numpy as np
import math
import cv2

def assemble(img_bgr, M):
    '''
    corners = []
    descriptors = []
    pairs = []
    M = []
    for i in range(len(img_bgr)):
        img_y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        corners.append(harris(img_y)) # corners format(y, x)
        descriptors.append(descriptor(img_y, corners[i]))
    for i in range(len(img_bgr)-1):
        pairs.append(find_pair(corners[i], descriptors[i], corners[i+1], descriptors[i+1]))
        M.append(solve_M(pairs[i][0], pairs[i][1]))
    '''
    rows, cols, ch = img_bgr[0].shape
    # Handling panorama boudaries
    x_max = cols
    y_min = 0
    y_max = rows
    
    position = []
    for i in range(len(img_bgr)):
#x = np.array(range(rows*cols))
#        y = np.array(range(rows*cols))
#       x = np.reshape(x, (rows, cols))
#       y = np.reshape(y, (rows, cols))
        x = np.mgrid[0:rows,0:cols][1]
        y = np.mgrid[0:rows,0:cols][0]
        if i == 0:
            position.append([y, x])
            continue
        new_x = x*M[i-1][0,0] + y*M[i-1][0,1] + 1*M[i-1][0,2]
        new_y = x*M[i-1][1,0] + y*M[i-1][1,1] + 1*M[i-1][1,2]
        print 'x', new_x
        print 'y', new_y
        new_x = new_x.astype(int)
        new_y = new_y.astype(int)
        if new_x[-1, -1] > x_max:
            x_max = new_x[-1, -1]
        if new_y[0, 0] < y_min:
            y_min = new_y[0, 0]
        if new_y[-1, -1] + rows > y_max:
            y_max = new_y[-1, -1]
        position.append([new_y, new_x])

    print 'boundary:', y_min, y_max, x_max
    panorama = np.zeros((y_max-y_min, x_max, 3))
    for i in range(len(img_bgr)):
        print 'processing img', i
        y_begin = int(position[i][0][0,0])
        y_end = y_begin + rows
        x_begin = int(position[i][1][0,0])
        x_end = x_begin + cols
        print 'y_begin:', y_begin
        print 'y_end:', y_end
        print 'x_begin:', x_begin
        print 'x_end:', x_end
        if i == 0:
            panorama[(y_begin-y_min):(y_end-y_min), x_begin:x_end, :] += img_bgr[i]
        else:
            if position[i][0][0,0] > position[i-1][0][0,0]:
                wy_begin = position[i][0][0,0]
                wy_end = position[i-1][0][-1,-1]
            elif position[i][0][0,0] < position[i-1][0][0,0]:
                wy_begin = position[i-1][0][0,0]
                wy_end = position[i][0][-1,-1]
            else:
                wy_begin = position[i][0][0,0]
                wy_end = position[i][0][-1,-1]
            wy_begin -= y_min
            wy_end -= y_min
            wx_begin = position[i][1][0,0]
            wx_end = position[i-1][1][0,-1]
            print 'wy', wy_begin, wy_end
            print 'wx', wx_begin, wx_end
            ny = wy_end - wy_begin + 1
            nx = wx_end - wx_begin + 1
            w1 = np.ones((y_max-y_min, x_max, 3))
            w2 = np.ones((rows, cols, 3))
            # mask_element = [0, 0.1, 0.2, ... 1]
            mask_element_1 = np.mgrid[1:0:complex(nx),0:3]
            mask_element_2 = np.mgrid[0:1:complex(nx),0:3]
            w1[wy_begin:wy_end+1, wx_begin:wx_end+1, :] = np.tile(mask_element_1[0],(ny,1,1))
            w2[(wy_begin-y_begin-y_min):(wy_end-y_begin-y_min+1), (wx_begin-x_begin):(wx_end-x_begin+1), :] = np.tile(mask_element_2[0],(ny,1,1)) 
            panorama *= w1
            print 'xx', x_begin, x_end 
            panorama[(y_begin-y_min):(y_end-y_min), x_begin:x_end, :] += img_bgr[i]*w2
#panorama[(y_begin-y_min):(y_end-y_min), x_begin:x_end, :] = \
#             panorama[(y_begin-y_min):(y_end-y_min), x_begin:x_end, :]*w1 + img_bgr[i]*w2
    cv2.imwrite('panorama.jpg', panorama) 


