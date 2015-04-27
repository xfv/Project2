import numpy as np
import math
import cv2

def assemble(img_bgr, M):
    rows, cols, ch = img_bgr[0].shape
    # Handling panorama boudaries
    x_max = cols
    y_min = 0
    y_max = rows
    for m in range(len(M)):
        M[m] = np.concatenate((M[m], np.array([0,0,1]).reshape(1,3)), 0) # Extend M to 3x3 matrix
    position = []
    for i in range(len(img_bgr)):
        # Define d_size for output img
        d_size_x = np.array([[0, cols], [0, cols]])
        d_size_y = np.array([[0, 0], [rows, rows]])
        d_size_x = d_size_x*M[i-1][0,0] + d_size_y*M[i-1][0,1] + 1*M[i-1][0,2]
        d_size_y = d_size_x*M[i-1][1,0] + d_size_y*M[i-1][1,1] + 1*M[i-1][1,2]
        if max(d_size_x[0,1], d_size_x[1,1]) > x_max:
            x_max = max(d_size_x[0,1], d_size_x[1,1])
        if min(d_size_y[0,0], d_size_y[0,1]) < y_min:
            y_min = min(d_size_y[0,0], d_size_y[0,1])
        if max(d_size_y[1,0], d_size_y[1,1]) > y_max:
            y_max = max(d_size_y[1,0], d_size_y[1,1])

    print 'boundary(y_min, y_max, x_min, x_max):', y_min, y_max, '0', x_max
    panorama = np.zeros((y_max-y_min+1, x_max+1, 3))
    for i in range(len(img_bgr)):
        print 'processing img', i
        x = np.mgrid[0:rows,0:cols][1]
        y = np.mgrid[0:rows,0:cols][0]
        if i == 0:
            print 'y_begin/end, x_begin/end:', y[0,0]-y_min, y[-1,-1]-y_min, x[0,0], x[-1,-1]
            panorama[(y[0,0]-y_min):(y[-1,-1]-y_min+1), x[0,0]:x[-1,-1]+1, :] += img_bgr[i]
            position.append([y[0,0]-y_min, y[-1,-1]-y_min, x[0,0], x[-1,-1]])
            continue
        
        d_size_x = np.array([[0, cols], [0, cols]])
        d_size_y = np.array([[0, 0], [rows, rows]])
        d_size_x = d_size_x*M[i-1][0,0] + d_size_y*M[i-1][0,1] + 1*M[i-1][0,2]
        d_size_y = d_size_x*M[i-1][1,0] + d_size_y*M[i-1][1,1] + 1*M[i-1][1,2]
        d_size_x = d_size_x.astype(int)
        d_size_y = d_size_y.astype(int)
        print d_size_x
        print d_size_y
        y_begin = min(d_size_y[0,0], d_size_y[0,1]) - y_min
        y_end = max(d_size_y[1,0], d_size_y[1,1]) - y_min
        x_begin = min(d_size_x[0,0], d_size_x[1,0])
        x_end = max(d_size_x[0,1], d_size_x[1,1])
        position.append([y_begin, y_end, x_begin, x_end])
        print 'y_begin/end, x_begin/end:', y_begin, y_end, x_begin, x_end
        M[i-1][0,2] -= x_begin
        M[i-1][1,2] -= y_begin
        d_size = (x_end-x_begin+1, y_end-y_begin+1)
#       x = cv2.warpPerspective(x.astype(float), M[i-1], d_size)
#       y = cv2.warpPerspective(y.astype(float), M[i-1], d_size)
        warp_img_bgr = np.zeros((y_end-y_begin+1, x_end-x_begin+1, 3))
        warp_img_bgr[:,:,0] = cv2.warpPerspective(img_bgr[i][:,:,0], M[i-1], d_size)
        warp_img_bgr[:,:,1] = cv2.warpPerspective(img_bgr[i][:,:,1], M[i-1], d_size)
        warp_img_bgr[:,:,2] = cv2.warpPerspective(img_bgr[i][:,:,2], M[i-1], d_size)
#      x = x.astype(int)
#       y = y.astype(int)

        if position[i][0] > position[i-1][0]:
            wy_begin = position[i][0]
            wy_end = position[i-1][1]
        else:
            wy_begin = position[i-1][0]
            wy_end = position[i][1]
        wx_begin = position[i][2]
        wx_end = position[i-1][3]
        print 'wy_begin/end, wx_begin/end:', wy_begin, wy_end, wx_begin, wx_end
        ny = wy_end - wy_begin + 1
        nx = wx_end - wx_begin + 1
        w1 = np.ones((y_max-y_min+1, x_max+1, 3))
        w2 = np.ones((y_end-y_begin+1, x_end-x_begin+1, 3))
        # mask_element = [0, 0.1, 0.2, ... 1]
        mask_element_1 = np.mgrid[1:0:complex(nx),0:3]
        mask_element_2 = np.mgrid[0:1:complex(nx),0:3]
        w1[wy_begin:wy_end+1, wx_begin:wx_end+1, :] = np.tile(mask_element_1[0],(ny,1,1))
        w2[(wy_begin-y_begin-y_min):(wy_end-y_begin-y_min+1), (wx_begin-x_begin):(wx_end-x_begin+1), :] = np.tile(mask_element_2[0],(ny,1,1)) 
        panorama *= w1
        panorama[y_begin:y_end+1, x_begin:x_end+1, :] += warp_img_bgr*w2
    cv2.imwrite('panorama.jpg', panorama) 


