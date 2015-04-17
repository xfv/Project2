import numpy
import cv2
import math
from harris import readFile


### get pixel value at point(row, col) on img
### no interpolation is used for now
### f = focal length
def getPixelValue(row, col, img, f):
    
    ### parameter
    s = f
    


    ### coordiante transformation 
    theta = math.atan(col/f) 
    h = row / math.sqrt( col*col + f*f )
    
    ### no interpolation for now
    ### Use interpolation if result is not good
    new_row = math.floor(s*h)
    new_col = math.floor(s*theta)

    ### check for boarders
    if new_row >= len(img):
        return numpy.zeros(img[0][0].shape) 

    if new_col >= len(img[0]):
        return numpy.zeros(img[0][0].shape) 


    return img[new_row][new_col]



### make cylinderal correction
### f = focal length
def cyCorrect(img, f):
    
    ### parameter

    
    ### dimension
    row_max = len(img)
    col_max = len(img[0])
    depth = len(img[0][0])

    ### get black boarder
    row_max += 20
    col_max += 20
    ### do the job
    result = numpy.zeros((row_max, col_max, depth))
    for row in range(row_max):
        for col in range(col_max):
            result[row][col] = getPixelValue(row, col, img, f)


    return result




### main function
if __name__ == "__main__":
    img_bgr = readFile('./sample/parrington/prtn13.jpg')
    img_result = cyCorrect(img_bgr, 904.289)
    cv2.imwrite('img_wrapped.jpg', img_result)
    




    
     
