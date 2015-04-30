import numpy
import cv2


### this function provides end-to-end alignment
### M_list is homography funcion of paired images
### The last M in M_list should be the paird of first image 
### and the last (match the first image with the last one)
### start coordinate of first image is (0, 0)
### the result shift each image and modify M_list
def drift(M_list):
    
    #result = M_list        ### this does not copy!
    #result = list(M_list)  ### this does not copy either!(only list but not numpy array)
    result = numpy.array(M_list, copy='true')   ### for iter still works!
    ### calculate drift
    ### initial point (0, 0, 1)
    origin = numpy.zeros((2, 1))
    ### note M is 3x3
    #origin = numpy.dot(M_list[-1], numpy.append(origin, [[1]], 0))
    #x_drift = origin[0]         ### total x_drift
    x_drift = 0
    
    for M in M_list:
        origin = numpy.dot(M, numpy.append(origin, [[1]], 0))
        x_drift += (numpy.dot(M, numpy.array([[0],[0],[1]])))[0]
    
    ### now origin is the point after homography of M_list
    y_drift = origin[1]     ### total drift
    x_coef = -y_drift/x_drift    ### spread total drift to x
    ### re-calculate M
    for R in result:
        ### y' = y + x_coef*x = m1x + m2y + m3 + x_coef*x 
        ###    = (m1+x_coef*x)x + m2y + m3
        R[1][0] = R[1][0] + x_coef

    tmp = []
    for i in range(len(result)):
        if i == 0:
            tmp.append(result[i])
        else:
            tmp.append(numpy.dot(tmp[i-1], result[i]))

    return tmp
        
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

    
    

