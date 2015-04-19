import harris
import numpy
import cv2
import projection
import matching


### main function for pararoma
print 'reading files..'
img_1 = harris.readFile('./sample/parrington/prtn13.jpg')
img_2 = harris.readFile('./sample/parrington/prtn12.jpg')


### correction
print 'cyCorrect'
img_1_cy = projection.cyCorrect(img_1, 704)
img_2_cy = projection.cyCorrect(img_2, 704)

### harris
### points are(x, y) not (row, col)
print 'harris'
points_1 = harris.harris(img_1_cy)
points_2 = harris.harris(img_2_cy)

### get feature
print 'getting features'
img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
feature_1 = matching.descriptor(img_1_gray, points_1)
feature_2 = matching.descriptor(img_2_gray, points_2)

print 'matching...'
pairs = matching.find_pair(points_1, feature_1, points_2, features_2)


### run RANSAC
### pairs = [ points_1, points_2 ]
print 'RANSAC...'
pairs = ransac.ransac(pairs)

img_1 = harris.drawDots(img_1, paris[0])
img_2 = harris.drawDots(img_2, paris[0])

cv2.imwrite('img_1.jpg', img_1)
cv2.imwrite('img_2.jpg', img_2)