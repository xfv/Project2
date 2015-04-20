import numpy
import matching
import cv2
import random




### Run RANSAC 
### point 1 and point 2 shoud be paired 
def ransac(points_1, points_2):

    ### parameters
    k = 4       ### only need four points to solve 2*2 matrix M
    n = 1000    ### iterations
    th = 700      ### threshold

    ### dimension
    len_1 = len(points_1)
    len_2 = len(points_2)



    ### main loop
    random.seed()
    inlier = numpy.array([])    ### this is the index of inliers
    selected_1 = numpy.zeros((k, 2))
    selected_2 = numpy.zeros((k, 2))

    for itr in range(n):

        ### pick k random feature points
        for i in range(k):
            selected_1[i] = points_1[random.randrange(len_1)]
            selected_2[i] = points_2[random.randrange(len_2)]
        
        ### calculate Homography for these four points
        homo = matching.solve_M(selected_1, selected_2)

        ### calculate distance and threshold
        point_solved = numpy.inner(points_2, homo)
        distance = numpy.sqrt( numpy.sum( (point_solved-points_1)**2, 1 ) )
        tmp_inlier = numpy.transpose( (distance<th).nonzero() )
        if len(tmp_inlier) > len(inlier):
            inlier = tmp_inlier


    print inlier

    result_1 = points_1[inlier]
    result_2 = points_2[inlier]

    result_1 = numpy.reshape(result_1, (len(result_1), 2))
    result_2 = numpy.reshape(result_2, (len(result_2), 2))

    return result_1, result_2 
        





