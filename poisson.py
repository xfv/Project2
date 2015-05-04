import numpy as np
import pyamg
import math
import cv2
import scipy.sparse

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
    y_begin, y_end, x_begin, x_end = position[0]
    ny_begin, ny_end, nx_begin, nx_end = position[1]
    mask_test = np.copy(panorama)
    mask_test[y_begin:y_end+1, x_begin:x_end+1, 0] = 255*warp_mask[0]
    mask_test[ny_begin:ny_end+1, nx_begin:nx_end+1, 1] = 255*warp_mask[1]
    #cv2.imshow('mask', mask_test)
    #cv2.waitKey(0)

    for i in range(len(img_bgr)):
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
                                                        * tmp_mask[y_begin:y_end, x_begin:x_end]
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
    pwarp_img_bgr = np.zeros(mask_xor[0].shape, dtype='float64')
    for i in range(len(img_bgr)):
        y_begin, y_end, x_begin, x_end = position[i]
        ### directly paste first image
        if i == 0:
            panorama[y_begin:y_end+1, x_begin:x_end+1, 0] = img_bgr[i][:, :, 0] * mask_xor[0]
            panorama[y_begin:y_end+1, x_begin:x_end+1, 1] = img_bgr[i][:, :, 1] * mask_xor[0]
            panorama[y_begin:y_end+1, x_begin:x_end+1, 2] = img_bgr[i][:, :, 2] * mask_xor[0]
            pwarp_img_bgr = img_bgr[i]
            pwarp_img_bgr = pwarp_img_bgr.astype('float64')
            continue


        ### d_size is the desired corp size
        d_size = (x_end-x_begin+1, y_end-y_begin+1)
        print 'warp img size:', d_size
        warp_img_bgr = np.zeros((y_end-y_begin+1, x_end-x_begin+1, 3))
        warp_img_bgr[:,:,0] = cv2.warpPerspective(img_bgr[i][:,:,0], H, d_size)
        warp_img_bgr[:,:,1] = cv2.warpPerspective(img_bgr[i][:,:,1], H, d_size)
        warp_img_bgr[:,:,2] = cv2.warpPerspective(img_bgr[i][:,:,2], H, d_size)

        ### directly paste image to panorama first
        panorama[y_begin:y_end+1, x_begin:x_end+1, 0] += warp_img_bgr[:, :, 0] * mask_xor[i]
        panorama[y_begin:y_end+1, x_begin:x_end+1, 1] += warp_img_bgr[:, :, 1] * mask_xor[i]
        panorama[y_begin:y_end+1, x_begin:x_end+1, 2] += warp_img_bgr[:, :, 2] * mask_xor[i]
        ### TEST
        cv2.imwrite('pano_mask'+ str(i) + '.jpg', panorama)
        '''
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
        '''
        ### calculate coordinate of overlay area
        ### no_prefix: current image
        ### p        : panorama
        if( position[i-1][0] > position[i][0] ):
            py_begin    = 0 
            y_begin     = position[i-1][0] - position[i][0]
        else:
            py_begin    = position[i][0] - position[i-1][0]
            y_begin     = 0
        if( position[i-1][1] > position[i][1] ):
            py_end      = position[i][1] - position[i-1][0]
            y_end       = position[i][1] - position[i][0]
        else:
            py_end      = position[i-1][1] -position[i-1][0]
            y_end       = position[i-1][1] - position[i][0]
        px_begin    = position[i][2] - position[i-1][2]
        px_end      = position[i-1][3] - position[i-1][2]
        x_begin     = 0
        x_end       = position[i-1][3] - position[i][2]

 
        ### poisson blending
        ### first cut the overlay block
        overlay_bgr = warp_img_bgr[y_begin:y_end, x_begin:x_end, :]
        print overlay_bgr.dtype
        poverlay_bgr = pwarp_img_bgr[py_begin:py_end, px_begin:px_end, :]
        print poverlay_bgr.dtype
        mask_overlay = mask_and[i-1][y_begin:y_end, x_begin:x_end]
        cv2.imshow('img', overlay_bgr.astype('uint8'))
        cv2.waitKey(0)
        cv2.imshow('img', poverlay_bgr.astype('uint8'))
        cv2.waitKey(0)
        tmp_overlay = np.copy(overlay_bgr)
        tmp_overlay[:, :, 0] *= mask_overlay
        tmp_overlay[:, :, 1] *= mask_overlay
        tmp_overlay[:, :, 2] *= mask_overlay
        cv2.imshow('img', tmp_overlay.astype('uint8'))
        cv2.waitKey(0)
        print 'shape'
        print mask_and[i-1].shape
        print position[i-1]
        print position[i]
        print overlay_bgr.shape, poverlay_bgr.shape, mask_overlay.shape

        ### performed on 3 channel seperately
        for c in range(3):
            overlay = overlay_bgr[:, :, c]
            poverlay = poverlay_bgr[:, :, c]
            
            rows = len(overlay)
            cols = len(overlay[0])
            ### fill in Ax = b and solve x
            print 'size'
            print rows, cols
            A = scipy.sparse.identity(rows*cols, format='lil')
            #A = np.zeros((rows*cols, rows*cols))
            #B = np.zeros(rows*cols) 
            P = pyamg.gallery.poisson(overlay.shape)
            B = P * overlay.flatten()
            ### fill in A
            for row in range(rows):
                #print row
                for col in range(cols):
                    #print col
                    Vpq = 0.0
                    Np = 0     ### neighbor count
                    neighbor = [0, 0, 0, 0] ### up, down, right, left
                    if(mask_overlay[row, col]):
                        fp = overlay[row, col]
                        gp = poverlay[row, col]
                        ### inside mask
                        ### get difference that is larger
                        if( row>0 ):
                            ### has up neighbor
                            fq = overlay[row-1, col]
                            gq = poverlay[row-1, col]
                            #if( mask_overlay[row-1, col] ):
                                #Vpq += (fp-fq, gp-gq)[(gp-gq)>(fp-fq)]
                                #Vpq += fp-fq
                            #else:
                                #Vpq += (fp-fq, gp-gq)[col<cols/2]
                                #Vpq += fp-fq
                            #Vpq += fp-fq
                            Vpq += gp-gq
                            neighbor[0] = -1
                            Np += 1
                        if( row<rows-1 ):
                            ### has down neighbor
                            fq = overlay[row+1, col]
                            gq = poverlay[row+1, col]
                            #Vpq += (fp-fq, gp-gq)[(gp-gq)>(fp-fq)]
                            #if( mask_overlay[row-1, col] ):
                                #Vpq += (fp-fq, gp-gq)[(gp-gq)>(fp-fq)]
                                #Vpq += fp-fq
                            #else:
                                #Vpq += (fp-fq, gp-gq)[col<cols/2]
                                #Vpq += fp-fq
                            #Vpq += fp-fq
                            Vpq += gp-gq
                            neighbor[1] = -1
                            Np += 1
                        if( col<cols-1 ):
                            ### has right neighbor
                            fq = overlay[row, col+1]
                            gq = poverlay[row, col+1]
                            #Vpq += (fp-fq, gp-gq)[(gp-gq)>(fp-fq)]
                            #if( mask_overlay[row-1, col] ):
                                #Vpq += (fp-fq, gp-gq)[(gp-gq)>(fp-fq)]
                                #Vpq += fp-fq
                            #else:
                                #Vpq += (fp-fq, gp-gq)[col<cols/2]
                                #Vpq += fp-fq
                            #Vpq += fp-fq
                            Vpq += gp-gq
                            neighbor[2] = -1
                            Np += 1
                        if( col>0 ):
                            ### has left neighbor
                            fq = overlay[row, col-1]
                            gq = poverlay[row, col-1]
                            #Vpq += (fp-fq, gp-gq)[(gp-gq)>(fp-fq)]
                            #if( mask_overlay[row-1, col] ):
                                #Vpq += (fp-fq, gp-gq)[(gp-gq)>(fp-fq)]
                                #Vpq += fp-fq
                            #else:
                                #Vpq += (fp-fq, gp-gq)[col/cols/2]
                                #Vpq += fp-fq
                            #Vpq += fp-fq
                            Vpq += gp-gq
                            neighbor[3] = -1
                            Np += 1
                        ### fill the coefficients
                        #print 'filling A'
                        Np = 4
                        neighbor = [-1, -1, -1, -1]
                        line = row*cols+col
                        A[line, row*cols+col] = Np 
                        A[line, (row-1)*cols+col] += neighbor[0]
                        A[line, (row+1)*cols+col] += neighbor[1]
                        A[line, row*cols+col+1] += neighbor[2]
                        A[line, row*cols+col-1] += neighbor[3]
                        #B[row*cols+col] = Vpq
                    else:
                        ### not inside mask
                        #A[row*cols+col] = 1
                        B[row*cols+col] = poverlay[row, col] 
                        #print 'else'

            print 'solving'
            A = A.tocsr()
            result = pyamg.solve(A, B, verb=True, tol=1e-8)
            print np.amax(result)
            result[result>255] = 255
            result[result<0] = 0
            #err, result = cv2.solve(A, B, None, cv2.DECOMP_SVD)       
            print py_begin-py_end, px_begin-px_end
            panorama[py_begin:py_end, px_begin:px_end, c] = np.reshape(result, (rows, cols))
        pwarp_img_bgr = warp_img_bgr
        cv2.imwrite('panorama'+str(i)+'.jpg', panorama) 
                       
                            
    cv2.imwrite('panorama.jpg', panorama) 
    return panorama


