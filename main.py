import harris
import numpy
import cv2 
import projection
import matching
import ransac


### main function for panorama
print 'reading files..'
img_1 = harris.readFile('./parrington/prtn13.jpg')
img_2 = harris.readFile('./parrington/prtn12.jpg')


### correction
print 'cyCorrect'
img_1_cy = projection.cyCorrect(img_1, 704.289)
img_2_cy = projection.cyCorrect(img_2, 704.0)
cv2.imwrite('img_1_cy.jpg', img_1_cy)
cv2.imwrite('img_2_cy.jpg', img_2_cy)

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
cv2.imwrite('img_1_harris.jpg', img_1_harris)
cv2.imwrite('img_2_harris.jpg', img_2_harris)

### get feature
print 'getting features...'
feature_1 = matching.descriptor(img_1_gray, points_1)
feature_2 = matching.descriptor(img_2_gray, points_2)


print 'matching...'
pairs = matching.find_pair_2(points_1, feature_1, points_2, feature_2)
print 'Got ', len(pairs[0]), ' pairs'
start = 0 
end = 250
print pairs[0][start:end], pairs[1][start:end]

img_1 = harris.drawDots(img_1_cy, pairs[0][start:end])
img_2 = harris.drawDots(img_2_cy, pairs[1][start:end])

cv2.imwrite('img_1_pair.jpg', img_1)
cv2.imwrite('img_2_pair.jpg', img_2)
img_match = matching.drawMatchLine(img_1, img_2, pairs[0][start:end], pairs[1][start:end])
cv2.imwrite('img_match_line.jpg', img_match)

### run RANSAC
### pairs = [ points_1, points_2 ]
print 'RANSAC...'
pairs = ransac.ransac(pairs[0], pairs[1])
print 'Got ', len(pairs[0]), ' pairs'
img_1 = harris.drawDots(img_1_cy, pairs[0][start:end])
img_2 = harris.drawDots(img_2_cy, pairs[1][start:end])
img_match = matching.drawMatchLine(img_1, img_2, pairs[0], pairs[1])
cv2.imwrite('img_match_line_2.jpg', img_match)


