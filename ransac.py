import numpy
import matching
import cv2
import random




### Run RANSAC 
### point 1 and point 2 shoud be paired 
def ransac(points_1, points_2):

    ### parameters
    k = 2       ### only need four points to solve 2*2 matrix M
    n = 15000    ### iterations
    th = 38     ### threshold

    ### dimension
    len_1 = len(points_1)
    len_2 = len(points_2)



    ### main loop
    random.seed()
    inlier = numpy.array([])    ### this is the index of inliers
    selected = numpy.zeros((k*2,2))
    selected_1 = numpy.zeros((k, 2))
    selected_2 = numpy.zeros((k, 2))

    for itr in range(n):

        ### pick k random feature points
        for i in range(k):
            rand = random.randrange(len_1)
            selected_1[i] = points_1[rand]
            selected_2[i] = points_2[rand]
        
        ### calculate Homography for these four points
        homo = matching.solve_M(selected_1, selected_2)

        ### calculate distance and threshold
        point_solved = numpy.inner(points_2, homo)
        distance = numpy.sum( (point_solved-points_1)**2, 1 ) 
        tmp_inlier = numpy.transpose( (distance<th).nonzero() )
        if len(tmp_inlier) > len(inlier):
            inlier = tmp_inlier
            selected = numpy.append(selected_1, selected_2, 0) 
            homo_match = homo


#print inlier

    result_1 = points_1[inlier]
    result_2 = points_2[inlier]

    result_1 = numpy.reshape(result_1, (len(result_1), 2))
    result_2 = numpy.reshape(result_2, (len(result_2), 2))
    
    print selected
    print homo_match
    print result_1, result_2

    return result_1, result_2 
        





