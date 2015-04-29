import harris
import numpy
import cv2 
import projection
import matching
import ransac
import assemble

### main function for panorama
print 'reading files..'
img = []
M = []
data_set = 17
for i in range(data_set,-1,-1):
    print 'loading', i
    img.append(harris.readFile('./parrington/prtn' + str(i).zfill(2) + '.jpg'))

for i in range(data_set):
    print i
    ### correction
    print 'cyCorrect'
    img_1 = img[i]
    img_2 = img[i+1]
    img_1_cy = projection.cyCorrect(img_1, 704.289)
    img_2_cy = projection.cyCorrect(img_2, 704.0)
    
    ### get gray scale
    img_1_gray = cv2.cvtColor(img_1_cy, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2_cy, cv2.COLOR_BGR2GRAY)
    
    ### harris
    ### points are(x, y) not (row, col)
    print 'harris'
    points_1 = harris.harris(img_1_gray)
    points_2 = harris.harris(img_2_gray)

    ### draw dots
    img_1_harris = harris.drawDots(img_1_cy, points_1)
    img_2_harris = harris.drawDots(img_2_cy, points_2)
    
    ### get feature
    print 'getting features...'
    feature_1 = matching.descriptor(img_1_gray, points_1)
    feature_2 = matching.descriptor(img_2_gray, points_2)
    
    print 'matching...'
    pairs = matching.find_pair(points_1, feature_1, points_2, feature_2)
    print 'Got ', len(pairs[0]), ' pairs'
#   start = 0 
#   end = 250
#   print pairs[0][start:end], pairs[1][start:end]
    
    ### run RANSAC
    ### pairs = [ points_1, points_2 ]
    print 'RANSAC...'
    pairs = ransac.ransac(pairs[0], pairs[1])
    tmp_M = matching.solve_M(pairs[0], pairs[1])
    tmp_M = numpy.concatenate((tmp_M, numpy.array([0,0,1]).reshape(1,3)), 0) # Extend M to 3x3 matrix
    if i == 0:
        M.append(tmp_M)
    else:
        print 'appending M', i, i-1
        print numpy.matrix(tmp_M)*numpy.matrix(M[i-1])
        M.append(numpy.matrix(tmp_M)*numpy.matrix(M[i-1]))
    print 'Got ', len(pairs[0]), ' pairs'

### run assemble
assemble.assemble(img, M)

