import numpy
import cv2
import math
from harris import readFile


### get pixel value at point(row, col) on img
### no interpolation is used for now
### f = focal length
### sample image set 1: f = 704 is ok
def getPixelValue(row, col, img, f):
    
    ### parameter
    s = f
    width = len(img[0])
    height = len(img)

    ### put coordinate center to (width/2, height/2)
    col = col - width/2 
    row = row - height/2 
    row = row - 10

    ### coordiante transformation 
    ### horizontal center should be at center of img
    new_col = f*math.tan(col/s)
    new_row = (row/s) * math.sqrt( new_col*new_col + f*f)
    
    ### no interpolation for now
    ### Use interpolation if result is not good
    new_col = math.floor(new_col) + width/2
    new_row = math.floor(new_row) + height/2

    ### check for boarders
    if (new_row >= height) or (new_row < 0) or (new_col >= width) or (new_col < 0):
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
    row_max += 40
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
    img_result = cyCorrect(img_bgr, 704.289)
    cv2.imwrite('img_wrapped.jpg', img_result)
    




    
     
