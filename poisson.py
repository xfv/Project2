import numpy as np
import math
import cv2

### poisson blending
### mostly copied from assemble
def poisson(img_bgr, M, mask):
    rows, cols, ch = img_bgr[0].shape
    # Handling panorama boudaries
    x_max = cols
    y_min = 0
    y_max = rows
    position = []
    warp_mask = []
    ### calculate img size of output
    ### d_size is current img border after homography
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
    ### panorama is initialized with full size
    panorama = np.zeros((y_max-y_min+1, x_max+1, 3))
    ### process each image
    ### no image is pasted at this step
    ### only calculate refined homography and mask
    for i in range(len(img_bgr)):
        print 'processing img', i
        ### calculate borders
        x = np.array([[0, cols-1], [0, cols-1]])
        y = np.array([[0, 0], [rows-1, rows-1]])
        if i == 0:
            #panorama[-y_min:rows-y_min, 0:cols, :] += img_bgr[i]
            position.append([-y_min, rows-y_min-1, 0, cols-1])
            warp_mask.append(mask)
            continue
        if i > 0:
            ### again d_size is the border of current image on absolute coordinate
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
        
        ### this is to match cv2.warpPerspective
        ### move the image after homography to top-left and corp to desired size
        H = np.copy(M[i-1])
        print 'Before:', H
        H[0,2] -= min(d_size_x[0,0], d_size_x[1,0])
        H[1,2] -= min(d_size_y[0,0], d_size_y[0,1])
        
        print 'After', H
        d_size = (x_end-x_begin+1, y_end-y_begin+1)
        warp_mask.append(cv2.warpPerspective(mask, H, d_size))
         
        
    mask_xor = [] ### mask for current image, starting from img0
    mask_and = [] ### mask for image overlay, starting from img1(overlay with 0)
    for i in range(len(img_bgr)):
        #y_begin, y_end, x_begin, x_end = position[i]
        #ny_begin, ny_end, nx_begin, nx_end = position[i+1]
        print 'img'+str(i)+'MASK'
        if i != len(img_bgr)-1:
        ### calculate coordinate of overlay area
            if( position[i][0] > position[i+1][0] ):
                y_begin = 0
                ny_begin = position[i][0] - position[i+1][0]
            else:
                y_begin = position[i+1][0] - position[i][0]
                ny_begin = 0
            if( position[i][1] > position[i+1][1] ):
                y_end = position[i+1][1] - position[i][0]
                ny_end = position[i+1][1] - position[i+1][0]
            else:
                y_end = position[i][1] - position[i][0] 
                ny_end = position[i][1] - position[i+1][0]
            x_begin = position[i+1][2] - position[i][2]
            x_end = position[i][3] - position[i][2]
            nx_begin = 0
            nx_end = position[i][3] - position[i+1][2]
            print x_end-x_begin, y_end-y_begin
            print nx_end-nx_begin, ny_end-ny_begin
            print 'end'
            ### calculate masks
            tmp_mask = np.copy(warp_mask[i]) ### mask for current image
            ntmp_mask = np.zeros((warp_mask[i+1].shape)) ### overlay mask for next image(relative to next image)
            ntmp_mask[ny_begin:ny_end, nx_begin:nx_end] = warp_mask[i+1][ny_begin:ny_end, nx_begin:nx_end]  \
                                                        * tmp_mask[y_begin:y_end, nx_begin:nx_end]
            mask_and.append(ntmp_mask)
            if i != 0:
                ### first image has no previous 
                tmp_mask -= mask_and[i-1]
            tmp_mask[y_begin:y_end, x_begin:x_end] -= ntmp_mask[ny_begin:ny_end, nx_begin:nx_end]
            mask_xor.append(tmp_mask)
        else:
            ### last image has no next image
            tmp_mask = np.copy(warp_mask[i])
            tmp_mask -= mask_and[i-1]
            mask_xor.append(tmp_mask)


    ### image blending 
    for i in range(len(img_bgr)):
        y_begin, y_end, x_begin, x_end = position[i]
        ### directly paste first image
        if i == 0:
            panorama[y_begin:y_end, x_begin:x_end, 0] = img_bgr[i][:, :, 0] * mask_xor[0]
            panorama[y_begin:y_end, x_begin:x_end, 1] = img_bgr[i][:, :, 1] * mask_xor[0]
            panorama[y_begin:y_end, x_begin:x_end, 2] = img_bgr[i][:, :, 2] * mask_xor[0]
            continue


        ### d_size is the desired corp size
        d_size = (x_end-x_begin+1, y_end-y_begin+1)
        print 'warp img size:', d_size
        warp_img_bgr = np.zeros((y_end-y_begin+1, x_end-x_begin+1, 3))
        warp_img_bgr[:,:,0] = cv2.warpPerspective(img_bgr[i][:,:,0], H, d_size)
        warp_img_bgr[:,:,1] = cv2.warpPerspective(img_bgr[i][:,:,1], H, d_size)
        warp_img_bgr[:,:,2] = cv2.warpPerspective(img_bgr[i][:,:,2], H, d_size)

        ### directly paste image to panorama first
        panorama[y_begin:y_end+1, x_begin:x_end+1, :] += warp_img_bgr * mask_and[i]

        ### calculate overlay border
        if position[i][0] > position[i-1][0]:
            wy_begin = y_begin 
        else:
            wy_begin = position[i-1][0]
        if position[i][1] > position[i-1][1]:
            wy_end = position[i-1][1]
        else:
            wy_end = position[i][1]
        wx_begin = position[i][2]
        wx_end = position[i-1][3]
        print 'wy_begin/end, wx_begin/end:', wy_begin, wy_end, wx_begin, wx_end
        ### width and height of overlay area
        ny = wy_end - wy_begin + 1
        nx = wx_end - wx_begin + 1

        ### poisson blending
        ### first cut the overlay block
        overlay_bgr = warp_img_bgr[wy_begin-y_begin:wy_end-y_begin, wx_begin:wx_end, :]
        poverlay_bgr = panorama[wy_begin:wy_end, wx_begin:wx_end, :] 
        mask_overlay = mask_and[i]

        ### performed on 3 channel seperately
        for c in range(3):
            overlay = overlay_bgr[:, :, c]
            poverlay = poverlay_bgr[:, :, c]
            
            rows = len(overlay)
            cols = len(overlay[0])
            ### fill in Ax = b and solve x
            A = np.zeros((rows*cols, rows*cols))
            B = np.zeros(rows*cols) 
            ### fill in A
            for row in range(rows):
                for col in range(cols):
                    Vpq = 0
                    Np = 0     ### neighbor count
                    neighbor = [0, 0, 0, 0] ### up, down, right, left
                    if(mask_overlay[row, col]):
                        fp = overlay[row, col]
                        gp = poverlay[row, col]
                        ### inside mask
                        ### get difference that is larger
                        if( row>0 and mask_overlay[row-1, col] ):
                            ### has up neighbor
                            fq = overlay[row-1, col]
                            gq = poverlay[row-1, col]
                            Vpq += (fp-fq, gp-gq)[(gp-gq)>(fp-fq)]
                            neighbor[0] = -1
                            Np += 1
                        if( row<rows-1 and mask_overlay[row+1, col] ):
                            ### has down neighbor
                            fq = overlay[row+1, col]
                            gq = poverlay[row+1, col]
                            Vpq += (fp-fq, gp-gq)[(gp-gq)>(fp-fq)]
                            neighbor[1] = -1
                            Np += 1
                        if( col<cols-1 and mask_overlay[row, col+1] ):
                            ### has right neighbor
                            fq = overlay[row, col+1]
                            gq = poverlay[row, col+1]
                            Vpq += (fp-fq, gp-gq)[(gp-gq)>(fp-fq)]
                            neighbor[2] = -1
                            Np += 1
                        if( col>0 and mask_overlay[row, col-1] ):
                            ### has left neighbor
                            fq = overlay[row, col-1]
                            gq = poverlay[row, col-1]
                            Vpq += (fp-fq, gp-gq)[(gp-gq)>(fp-fq)]
                            neighbor[3] = -1
                            Np += 1
                        ### fill the coefficients
                        A[row*cols+col] = Np
                        A[(row-1)*cols+col] += neighbor[0]
                        A[(row+1)*cols+col] += neighbor[1]
                        A[row*cols+col+1] += neighbor[2]
                        A[row*cols+col-1] += neighbor[3]
                        B[row*cols+col] = Vpq
                    else:
                        ### not inside mask
                        A[row*cols+col] = 1
                        B[row*cols+col] = 0

            err, result = cv2.solve(A, B, None, cv2.DECOMP_SVD)       
            panorama[wy_begin:wy_end, wx_begin:wx_end, c] += result 
        cv2.imwrite('panorama'+str(i)+'.jpg', panorama) 
                       
                            
    cv2.imwrite('panorama.jpg', panorama) 
    return panorama


