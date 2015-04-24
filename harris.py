import numpy
import cv2


### read image file here
def readFile(path):
    img_bgr = cv2.imread(path)
    return img_bgr

### draw dots on img
### default dot color is red
### have to make copy to prevent drawing on original image
def drawDots(img, dots):
    tmp = img.copy()
    for pt in range(len(dots)):
        point = (dots[pt][0], dots[pt][1])
        cv2.circle(tmp, point, 1, (0, 0, 255), -1)

    return tmp

### check border
def checkNeighbor(points, x_max, y_max):
    result = numpy.copy(points)
    idx_del = []
    ### percentage of frame valid
    range_x = 0.8
    range_y = 0.8

    x_min = x_max * (1-range_x) / 2
    y_min = y_max * (1-range_y) / 2
    ### collect points too near to border
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        if x<x_min or y<y_min or x>x_max*(1+range_x)/2 or y>y_max*(1+range_y)/2:
            idx_del.append(i)
    ##        print 'delete point ', x, y

    ### delete points
    result = numpy.delete(result, idx_del, 0)

    return result


### Harris Corner Detection
def harris(img_y):

    ### parameters

    sigma   = 0.05       ### sigma for calculating pixel sum
    k       = 0.04      ### empirical constant 0.04-0.06
    R_min   = 100000       ### threshold for harris corner detector
    dot_max = 600       ### maximun corners

    ### need gray scale only
    #img_y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


    ### Calculate x and y derivatives of img
    ### cv2.Sobel already add Gaussian smoothing
    ### to calculate derivatives without smoothing, use cv2.Sobel(src, depth, x_ord, y_ord, dst, 1) instead
    ddepth = cv2.CV_32F
    #ddepth = -1
    Ix = cv2.Sobel(img_y, ddepth, 1, 0, None, 1)
    Iy = cv2.Sobel(img_y, ddepth, 0, 1, None, 1)

    ### Calculate Ixx, Iyy, Ixy
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    ### Free memory
    del Ix
    del Iy

    ### Calculate pixel sum of Ixx, Iyy, Ixy with noise filtering
    Sxx = cv2.GaussianBlur(Ixx, (3, 3), sigma)
    Syy = cv2.GaussianBlur(Iyy, (3, 3), sigma)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), sigma)

    ### Free memory again
    del Ixx
    del Iyy
    del Ixy

    ### Calculate R
    ### Calculation of M is skipped since we only need det(M) and trace(M)
    det     = (Sxx * Syy) - Sxy**2
    trace   = Sxx + Syy 
    R = det - k*(trace**2)
    R = numpy.abs(R)


    ### delete point by finding local maximum
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(12,12))
    R_dilate = cv2.dilate(R, kernel) 
    R[R!=R_dilate] = 0          ### remove point that are not local maximum
    
    ### Free memory
    del det, R_dilate
    del trace


    ### Find corners 
    ### first use max corners(neglect R_min)
    R_expand = numpy.ravel(R)
    corners = numpy.argpartition(R_expand, -dot_max)[-dot_max:]
    ### corners now is 1-D, not sorted(no need to sort)
    ### now use corner[dot_max] to check if points are sufficient
    ### if R[corner] > R_min then points are enough
    ### if not, then no dot_max points can be found
    cols = len(R[0])
    rows = len(R)
    row_corner = (corners[0])/cols
    col_corner = corners[0] - row_corner*cols
    if R[row_corner][col_corner] < R_min:
        ### this means we should use R_min to get corners
        corners = numpy.transpose((R>R_min).nonzero())
        corners = numpy.fliplr(corners)         ### (row, column) = (y, x) so need to switch 
        print 'harris points not enough... max = ', dot_max, ', got ', len(corners)
        corners = checkNeighbor(corners, len(img_y[0]), len(img_y) )
    else:
        corner_row = corners/cols 
        corner_col = corners - corner_row*cols 
        corners = numpy.array([corner_col, corner_row]) ### no need to flip since (col, row) = (x, y)
        corners = numpy.swapaxes(corners, 0, 1)
        corners = checkNeighbor(corners, len(img_y[0]), len(img_y))


    ### Generate harris corner image by opencv function
    ### This is for comparison
    ### GoodFeaturesToTrack
    corners_good = cv2.goodFeaturesToTrack(img_y, 500, 0.0005, 10, None,  None, 3, True, k)
    corners_good = numpy.reshape(corners_good, (len(corners_good), 2) )

    
    return corners 




     

    


### main function
if __name__ == "__main__":
    img_bgr = readFile('./sample/cow.jpg')
    img_result = harris(img_bgr)
    cv2.imwrite('corner_harris.jpg', img_result)



