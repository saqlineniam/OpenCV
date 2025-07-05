import cv2 as cv
import numpy as np



img = cv.imread('Photos/cats 2.jpg')
cv.imshow('cats', img)

blank = np.zeros(img.shape[:2], dtype = 'uint8')
cv.imshow('Blank', blank)


rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2),100, 255, -1)

Weird_shape = cv.bitwise_and(rectangle, circle)
cv.imshow('Weird_shaped mask image', Weird_shape)


masked = cv.bitwise_and(img, img, mask = Weird_shape)
cv.imshow('Masked', masked)






cv.waitKey(0)