import harris
import numpy
import cv2 
import projection
import matching
import ransac
from assemble2 import assemble_2
from poisson import poisson
from drift import drift

### main function for panorama
print 'reading files..'
img = []
img_cy = []
M = []
data_set = 16 
focal = 470.0
for i in range(data_set,-1,-1):
    print 'loading', i
    #img_read = harris.readFile('./parrington/prtn' + str(i).zfill(2) + '.jpg')
    img_read = harris.readFile('./photo/dataset/6/small/img' + str(i).zfill(2) + '.jpg')
    
    img.append(img_read)
    ### correction
    print 'cyCorrect', i
    img_cy.append(projection.cyCorrect(img_read, focal))

for i in range(data_set):
    print i
    img_1 = img[i]
    img_2 = img[i+1]
    img_1_cy = img_cy[i]
    img_2_cy = img_cy[i+1]
    
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
    start = 0 
    end = 250
    
    ### run RANSAC
    ### pairs = [ points_1, points_2 ]
    print 'RANSAC...'
    pairs = ransac.ransac(pairs[0], pairs[1])
    #img_matching = matching.drawMatchLine(img_1_cy, img_2_cy, pairs[0][start:end], pairs[1][start:end])
    #cv2.imwrite('match_line'+str(i)+'.jpg', img_matching )
    tmp_M = matching.solve_M(pairs[0], pairs[1])
    tmp_M = numpy.concatenate((tmp_M, numpy.array([0,0,1]).reshape(1,3)), 0) # Extend M to 3x3 matrix
    if i == 0:
        M.append(tmp_M)
    else:
        print 'appending M', i, i-1
        M.append( numpy.dot(M[i-1], tmp_M) )
    print 'Got ', len(pairs[0]), ' pairs'



### run poisson blending
### get mask
print img[0].shape
mask = numpy.ones((len(img[0]), len(img[0][0]), 3)) 
mask = projection.cyCorrect(mask, focal)[:, :, 0]
print mask.shape
### run assemble
img_pano_linear = assemble_2(img_cy, M, mask)
img_pano_linear = drift(img_pano_linear, M[-1])
cv2.imwrite('pano_linear.jpg', img_pano_linear)
### run poisson
img_pano_poisson = poisson(img_cy, M, mask)
img_pano_poisson = drift(img_pano_poisson, M[-1])
cv2.imwrite('pano_poisson.jpg', img_pano_poisson)



