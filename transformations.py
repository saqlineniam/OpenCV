import cv2 as cv
import numpy as np

img = cv.imread('Photos/park.jpg')
cv.imshow('park', img)

#translation
def translation(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x  ---> left
# x ---> right
# -y ---> up
# y ---> down

translated = translation(img,100,100)
cv.imshow('Translated', translated)

#rotation
def rotate(image, angle, rotPoint = None):
    (height,width) = image.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2, height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(image, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)

rotated_rotated = rotate(rotated, -45)
cv.imshow('Rotated Rotated', rotated_rotated)

#resize
resized = cv.resize(img,(500,500),interpolation = cv.INTER_CUBIC)
cv.imshow('Resized', resized)

#Flipping
flipped = cv.flip(img, -1)
cv.imshow('Flipped', flipped)

#cropping
cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
