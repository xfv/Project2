import numpy
import cv2


### this function provides end-to-end alignment
### M is homography funcion of last image to first
### start coordinate of first image is (0, 0)
### the result shift image 
def drift(img, M):
    
    ### calculate drift
    origin = numpy.array([[0], [0], [1]])
    origin = numpy.dot(M, origin)
    drift_x = origin[0]
    drift_y = origin[1]

    x_coeff = -drift_y/drift_x
    homo = numpy.diag(numpy.ones(3))
    ### fill in coeffiecients
    ###     1   0   0
    ### x_coeff 1   0
    ###     0   0   1   
    homo[1, 0] = x_coeff
    print homo
    print img.shape

    result = cv2.warpPerspective(img, homo, (len(img[0]), len(img)))
    return result 
### main function
if __name__ == "__main__":
    M1 = numpy.array([[1, 0, 10], [0, 1, 1]], dtype='float64')
    M2 = numpy.array([[1, 0, 10], [0, 1, 1]], dtype='float64')
    M3 = numpy.array([[1, 0, 10], [0, 1, 1]], dtype='float64')
    M_list = [M1, M2, M3]
    M_list = list(drift(M_list))

    print M_list
    origin = numpy.zeros((2,1))
    for M in M_list:
        origin = numpy.dot(M, numpy.append(origin, [[1]], 0))
    print origin

    
    

