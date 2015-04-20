import harris
import numpy
import cv2 
import projection
import matching
import ransac


### main function for panorama
print 'reading files..'
img_1 = harris.readFile('parrington/prtn13.jpg')
img_2 = harris.readFile('parrington/prtn12.jpg')


### correction
print 'cyCorrect'
img_1_cy = projection.cyCorrect(img_1, 704.289)
img_2_cy = projection.cyCorrect(img_2, 704.0)
cv2.imwrite('img_1_cy.jpg', img_1_cy)
cv2.imwrite('img_2_cy.jpg', img_2_cy)
### get gray scale
img_1_gray = cv2.cvtColor(img_1_cy, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(img_2_cy, cv2.COLOR_BGR2GRAY)
cv2.imwrite('img_1_gray.jpg', img_1_gray)
cv2.imwrite('img_2_gray.jpg', img_2_gray)

### harris
### points are(x, y) not (row, col)
print 'harris'
points_1 = harris.harris(img_1_gray)
points_2 = harris.harris(img_2_gray)
### draw dots
img_1_harris = harris.drawDots(img_1_cy, points_1)
img_2_harris = harris.drawDots(img_2_cy, points_2)
cv2.imwrite('img_1_harris.jpg', img_1_harris)
cv2.imwrite('img_2_harris.jpg', img_2_harris)

### get feature
print 'getting features...'
feature_1 = matching.descriptor(img_1_gray, points_1)
feature_2 = matching.descriptor(img_2_gray, points_2)

print 'matching...'
pair_1, pair_2 = matching.find_pair(points_1, feature_1, points_2, feature_2)

img_1_pair = harris.drawDots(img_1_cy, pair_1)
img_2_pair = harris.drawDots(img_2_cy, pair_2)
cv2.imwrite('img_1_pair.jpg', img_1_pair)
cv2.imwrite('img_2_pair.jpg', img_2_pair)

### run RANSAC
### pairs = [ points_1, points_2 ]
print 'RANSAC...'
pairs = ransac.ransac(pair_1, pair_2)

#print pairs[0].shape

