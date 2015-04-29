import numpy as np
import math
import cv2

def assemble(img_bgr, M):
    rows, cols, ch = img_bgr[0].shape
    # Handling panorama boudaries
    x_max = cols
    y_min = 0
    y_max = rows
    position = []
    for i in range(1, len(img_bgr)):
        # Define d_size for output img
        x = np.array([[0, cols], [0, cols]])
        y = np.array([[0, 0], [rows, rows]])
        d_size_x = x*M[i-1][0,0] + y*M[i-1][0,1] + 1*M[i-1][0,2]
        d_size_y = x*M[i-1][1,0] + y*M[i-1][1,1] + 1*M[i-1][1,2]
        d_size_x = d_size_x.astype(int)
        d_size_y = d_size_y.astype(int)
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
        x = np.array([[0, cols-1], [0, cols-1]])
        y = np.array([[0, 0], [rows-1, rows-1]])
        if i == 0:
            panorama[-y_min:rows-y_min, 0:cols, :] += img_bgr[i]
            position.append([-y_min, rows-y_min-1, 0, cols-1])
            continue
        if i > 0:
            d_size_x = x*M[i-1][0,0] + y*M[i-1][0,1] + 1*M[i-1][0,2]
            d_size_y = x*M[i-1][1,0] + y*M[i-1][1,1] + 1*M[i-1][1,2]
        y_begin = int(min(d_size_y[0,0], d_size_y[0,1]) - y_min)
        y_end = int(max(d_size_y[1,0], d_size_y[1,1]) - y_min)
        x_begin = int(min(d_size_x[0,0], d_size_x[1,0]))
        x_end = int(max(d_size_x[0,1], d_size_x[1,1]))
        print 'y_begin/end, x_begin/end:', y_begin, y_end, x_begin, x_end
        position.append([y_begin, y_end, x_begin, x_end])
        
        print d_size_x
        print d_size_y
        print 'dx', min(d_size_x[0,0], d_size_x[1,0])
        print 'dy', min(d_size_y[0,0], d_size_y[0,1])
        
        H = np.copy(M[i-1])
        print 'Before:', H
        H[0,2] -= min(d_size_x[0,0], d_size_x[1,0])
        H[1,2] -= min(d_size_y[0,0], d_size_y[0,1])
        
        print 'After', H
         
        test_x = np.array([[0, cols-1], [0, cols-1]])
        test_y = np.array([[0,0], [rows-1, rows-1]])
        ttest_x = test_x*H[0,0] + test_y*H[0,1] + 1*H[0,2]
        ttest_y = test_x*H[1,0] + test_y*H[1,1] + 1*H[1,2]
        print 'warp x idx:', ttest_x
        print 'warp y idx:', ttest_y
        
        d_size = (x_end-x_begin+1, y_end-y_begin+1)
        print 'warp img size:', d_size
        warp_img_bgr = np.zeros((y_end-y_begin+1, x_end-x_begin+1, 3))
        warp_img_bgr[:,:,0] = cv2.warpPerspective(img_bgr[i][:,:,0], H, d_size)
        warp_img_bgr[:,:,1] = cv2.warpPerspective(img_bgr[i][:,:,1], H, d_size)
        warp_img_bgr[:,:,2] = cv2.warpPerspective(img_bgr[i][:,:,2], H, d_size)

        if position[i][0] > position[i-1][0]:
            wy_begin = position[i][0]
        else:
            wy_begin = position[i-1][0]
        if position[i][1] > position[i-1][1]:
            wy_end = position[i-1][1]
        else:
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
        w2[(wy_begin-y_begin):(wy_end-y_begin+1), (wx_begin-x_begin):(wx_end-x_begin+1), :] = np.tile(mask_element_2[0],(ny,1,1)) 
        panorama *= w1
        cv2.imshow('warp img', warp_img_bgr)
        cv2.waitKey(0)
        panorama[y_begin:y_end+1, x_begin:x_end+1, :] += warp_img_bgr*w2
    cv2.imwrite('panorama.jpg', panorama) 


